use std::collections::{HashMap, HashSet};
use std::convert::{From, Into, TryInto};
use std::iter;

use crate::spatial::predicates::{self, Orient};
use crate::spatial::Point;
use crate::visual::Visualizer;

use super::bounds::{self, PlaneBoundary, Segment};
use super::MeshError;

// enum for vertex indices in the ghost vertex scheme
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum VIdx {
    Real(usize),
    Bound(bounds::VIdx),
    Ghost,
}

impl VIdx {
    fn is_ghost(&self) -> bool {
        match self {
            VIdx::Ghost => true,
            _ => false,
        }
    }
    fn is_real(&self) -> bool {
        match self {
            VIdx::Real(_) => true,
            _ => false,
        }
    }
    fn is_bound(&self) -> bool {
        match self {
            VIdx::Bound(_) => true,
            _ => false,
        }
    }
}

// structs for ordered pairs / triplets of vertex indices
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Triangle(VIdx, VIdx, VIdx);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Edge(VIdx, VIdx);

impl Triangle {
    fn edges(self) -> [Edge; 3] {
        [
            Edge(self.0, self.1),
            Edge(self.1, self.2),
            Edge(self.2, self.0),
        ]
    }

    fn is_ghost(self) -> bool {
        self.0.is_ghost() || self.1.is_ghost() || self.2.is_ghost()
    }

    fn ghost_count(self) -> usize {
        let mut res = 0;
        if self.0.is_ghost() {
            res += 1;
        }
        if self.1.is_ghost() {
            res += 1;
        }
        if self.2.is_ghost() {
            res += 1;
        }
        res
    }
}

impl From<(VIdx, VIdx, VIdx)> for Triangle {
    fn from(t: (VIdx, VIdx, VIdx)) -> Triangle {
        Triangle(t.0, t.1, t.2)
    }
}

impl Edge {
    fn rev(self) -> Edge {
        Edge(self.1, self.0)
    }

    fn is_ghost(self) -> bool {
        self.0.is_ghost() || self.1.is_ghost()
    }
}

impl From<(VIdx, VIdx)> for Edge {
    fn from(e: (VIdx, VIdx)) -> Edge {
        Edge(e.0, e.1)
    }
}

impl From<Segment> for Edge {
    fn from(s: Segment) -> Edge {
        Edge(VIdx::Bound(s.0), VIdx::Bound(s.1))
    }
}

pub struct PlaneTriangulation {
    // mesh bounds structure
    bound: PlaneBoundary,

    // vector of vertex locations
    vertices: Vec<(f64, f64)>,

    // map with three entries per triangle for fast lookup
    tris: HashMap<Edge, VIdx>,

    // structures for readout / random selection
    // TODO maybe combine tri_idxs with tris for space efficiency
    tri_idxs: HashMap<Edge, usize>,
    tri_list: Vec<Triangle>,

    // record of coappearing vertices in recently added triangles
    recents: HashMap<VIdx, VIdx>,
}

impl PlaneTriangulation {
    pub fn new(bound: PlaneBoundary) -> Self {
        // initialize a new empty triangulation with the given boundary
        PlaneTriangulation {
            bound,
            vertices: Vec::new(),
            tris: HashMap::new(),
            tri_idxs: HashMap::new(),
            tri_list: Vec::new(),
            recents: HashMap::new(),
        }
    }

    fn store_vertex<T: TryInto<(f64, f64)>>(&mut self, p: T) -> VIdx {
        // store a new vertex, returning its index
        if let Ok((x, y)) = p.try_into() {
            self.vertices.push((x, y));
            VIdx::Real(self.vertices.len() - 1)
        } else {
            panic!("bad vertex input");
        }
    }

    fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    fn get(&self, v: VIdx) -> Option<(f64, f64)> {
        // retrieve a vertex
        match v {
            VIdx::Real(i) => self.vertices.get(i).cloned(),
            VIdx::Bound(i) => self.bound.get(i),
            VIdx::Ghost => None,
        }
    }

    fn all_vidx(&self) -> impl Iterator<Item = VIdx> {
        // return an iterator over all valid VIdxs
        (0..self.vertices.len())
            .map(|x| VIdx::Real(x))
            .chain(self.bound.all_vidx().map(|x| VIdx::Bound(x)))
            .chain(iter::once(VIdx::Ghost))
    }

    fn get_triangle_points(&self, tri: Triangle) -> Option<(Point, Point, Point)> {
        // retrieve a triangle as a tuple of Points
        if tri.is_ghost() {
            return None;
        }
        let Triangle(u, v, w) = tri;
        Some((
            self.get(u).unwrap().into(),
            self.get(v).unwrap().into(),
            self.get(w).unwrap().into(),
        ))
    }

    fn in_circle<T: Into<Triangle>>(&self, tri: T, x: VIdx) -> bool {
        // determine whether x lies in the oriented triangle tri's circumcircle
        let tri = tri.into();

        // end immediately for doubly-ghost edge cases
        if tri.ghost_count() > 1 {
            return false
        }

        // first, if tri is a ghost triangle, only check if x is to the left of the non-ghost segment
        let ghosts = (tri.0.is_ghost(), tri.1.is_ghost(), tri.2.is_ghost());
        if ghosts.0 {
            return self.triangle_dir((tri.1, tri.2, x)).unwrap() != Orient::Negative;
        }
        if ghosts.1 {
            return self.triangle_dir((tri.2, tri.0, x)).unwrap() != Orient::Negative;
        }
        if ghosts.2 {
            return self.triangle_dir((tri.0, tri.1, x)).unwrap() != Orient::Negative;
        }

        // a ghost vertex is never inside a real triangle
        if x.is_ghost() {
            return false
        }

        // if tri is a proper triangle, use the in_circle spatial predicate
        // sanity check: tri should be correctly oriented (TODO remove this check)
        assert_eq!(self.triangle_dir(tri), Some(Orient::Positive));

        println!("checking in_circle for {:?}, {:?}", self.get(x).unwrap(), self.get_triangle_points(tri).unwrap());

        predicates::in_circle(
            self.get(x).expect("nonexistent vidx").into(),
            self.get_triangle_points(tri).unwrap(),
        )
    }

    fn triangle_dir<T: Into<Triangle>>(&self, tri: T) -> Option<Orient> {
        // determine the orientation of a triangle
        self.get_triangle_points(tri.into())
            .map(predicates::triangle_dir)
    }

    fn to_left<E: Into<Edge>>(&self, e: E, p: VIdx) -> bool {
        // determine whether p is to the left of ('in front of') e
        // in constrast to triangle_dir, this properly handles ghost points
        // TODO make sure this is in fact the right ghost handling
        let e = e.into();
        if p.is_ghost() || e.is_ghost() {
            true
        } else {
            self.triangle_dir((e.0, e.1, p)) == Some(Orient::Positive)
        }
    }

    fn add_triangle<T: Into<Triangle>>(&mut self, tri: T) -> Result<(), MeshError> {
        // add a triangle to the triangulation
        // only one of the vertices may be a ghost vertex
        // if no vertices are ghosts, as a sanity check ensure the triangle is positive
        let tri = tri.into();
        match tri.ghost_count() {
            2 | 3 => return Err(MeshError::triangle("doubly / triply ghost triangle")),
            0 => {
                if self.triangle_dir(tri) != Some(Orient::Positive) {
                    return Err(MeshError::triangle("negative triangle addition requested"));
                }
            }
            1 => (),
            _ => unreachable!(),
        }

        // a new triangle may not share an (oriented) edge with an old triangle
        let edges = tri.edges();
        if edges.iter().any(|e| self.tris.contains_key(e)) {
            return Err(MeshError::triangle("existing triangle conflicts with edge"));
        }

        // triangle is ok to add - add it
        self.tris.insert(edges[0], tri.2);
        self.tris.insert(edges[1], tri.0);
        self.tris.insert(edges[2], tri.1);

        let idx = self.tri_list.len();
        for e in edges.iter().cloned() {
            self.tri_idxs.insert(e, idx);
        }
        self.tri_list.push(tri);

        // add to the recent co-appearing points map for later adjacent_one use
        // seems reasonable to not insert any ghost pairs for adjacent_one, right?
        for e in edges.iter().cloned() {
            if !e.is_ghost() {
                self.recents.insert(e.0, e.1);
                self.recents.insert(e.1, e.0);
            }
        }

        Ok(())
    }

    fn delete_triangle<T: Into<Triangle>>(&mut self, tri: T) -> Result<(), MeshError> {
        // delete a triangle
        let tri = tri.into();
        let edges = tri.edges();
        if self.tris.get(&edges[0]) == Some(&tri.2) {
            // store the idnex of the removed triangle and remove entries
            let idx = self.tri_idxs.get(&edges[0]).unwrap().clone();
            for e in edges.iter() {
                self.tris.remove(e);
                self.tri_idxs.remove(e);
            }

            // do a swap to avoid O(n) removal
            // TODO address edge case where the last triangle is removed
            let to_move = self.tri_list.pop().unwrap();
            self.tri_list[idx] = to_move;

            // update the index of the moved triangle
            for e in to_move.edges().iter().cloned() {
                self.tri_idxs.insert(e, idx);
            }
            Ok(())
        } else {
            Err(MeshError::triangle("triangle to be deleted does not exist"))
        }
    }

    fn adjacent<E: Into<Edge>>(&self, e: E) -> Option<VIdx> {
        // return Some(w) if the positively oriented uvw exists
        self.tris.get(&e.into()).cloned()
    }

    fn adjacent_one(&self, u: VIdx) -> Option<Edge> {
        // return an arbitrary triangle including u, if one exists
        // if u has been part of a recent triangle, return it
        // warning, may return a ghost triangle
        if let Some(v) = self.recents.get(&u).cloned() {
            let e = Edge(u, v);
            if let Some(w) = self.tris.get(&e).cloned() {
                return Some(Edge(v, w));
            }
            if let Some(w) = self.tris.get(&e.rev()).cloned() {
                return Some(Edge(w, v));
            }
        }

        // otherwise, search everything
        for i in 0..self.vertex_count() {
            let v = VIdx::Real(i);
            let e = Edge(u, v);
            if let Some(w) = self.tris.get(&e).cloned() {
                return Some((v, w).into());
            }
            if let Some(w) = self.tris.get(&e.rev()).cloned() {
                return Some((w, v).into());
            }
        }

        // check if there's a ghost triangle
        if let Some(w) = self.tris.get(&(u, VIdx::Ghost).into()).cloned() {
            return Some((VIdx::Ghost, w).into());
        }
        if let Some(w) = self.tris.get(&(VIdx::Ghost, u).into()).cloned() {
            return Some((w, VIdx::Ghost).into());
        }

        None
    }

    fn bowyer_watson_dig(&mut self, u: VIdx, v: VIdx, w: VIdx) {
        // u is a new vertex
        // check if uvw is delaunay
        if let Some(x) = self.adjacent((w, v)) {
            if self.in_circle((u, v, w), x) {
                self.delete_triangle((w, v, x));
                self.bowyer_watson_dig(u, v, x);
                self.bowyer_watson_dig(u, x, w);
            } else {
                self.add_triangle((u, v, w))
                    .expect("could not add triangle");
            }
        }
    }

    fn bowyer_watson_insert<T: Into<Triangle>>(&mut self, u: VIdx, tri: T) {
        // insert a vertex into a delaunay triangulation, maintaining the delaunay property
        // tri is a triangle whose cirmcumcircle encloses u
        let Triangle(v, w, x) = tri.into();
        self.delete_triangle((v, w, x))
            .expect("could not delete triangle");
        self.bowyer_watson_dig(u, v, w);
        self.bowyer_watson_dig(u, w, x);
        self.bowyer_watson_dig(u, x, v);
    }

    fn gift_wrap_finish(&mut self, e: Edge) -> Option<(Triangle, Edge, Edge)> {
        // finish a triangle from a directed edge
        // return the triangle and the two new directed edges

        // TODO include visibility algorithms for the constrained triangulation

        let mut tri: Option<Triangle> = None;
        for v in self.all_vidx() {
            // do not construct doubly-ghost triangles
            if e.is_ghost() && v.is_ghost() {
                continue;
            }
            // proceed if v is in front of e and tri is either None or encircling
            if self.to_left(e, v) && tri.map_or(true, |t| self.in_circle(t, v)) {
                // TODO add constrained visibility check here
                tri = Some((e.0, e.1, v).into());
            }
        }

        // return the triangle and the two new directed edges
        tri.map(|t| (t, (t.1, t.2).into(), (t.2, t.0).into()))
    }

    pub fn gift_wrap(&mut self) {
        // gift-wrap the triangulation from scratch
        self.tris.clear();
        self.tri_idxs.clear();
        self.tri_list.clear();
        self.recents.clear();

        let mut to_finish: HashSet<Edge> = HashSet::new();

        for e in self.bound.all_walls().into_iter().map(|s| s.into()) {
            to_finish.insert(e);
        }

        while !to_finish.is_empty() {
            // remove an element from the set
            let e = to_finish.iter().next().cloned().unwrap();
            to_finish.remove(&e);

            if let Some((tri, a, b)) = self.gift_wrap_finish(e) {
                self.add_triangle(tri).expect("could not add triangle");
                if !to_finish.remove(&a) {
                    // if a was not in set, add its reverse
                    to_finish.insert(a.rev());
                }
                if !to_finish.remove(&b) {
                    to_finish.insert(b.rev());
                }
            }
        }

        let mut vis = self.visualize();
        vis.draw(format!("test_generated/eek.png").as_str(), ());
    }

    pub fn visualize(&self) -> Visualizer {
        let vidxs: Vec<VIdx> = self.all_vidx().filter(|x| !x.is_ghost()).collect();

        let nodes: Vec<Point> = vidxs.iter().map(|x| self.get(*x).unwrap().into()).collect();

        let idx_map: HashMap<VIdx, usize> = vidxs
            .iter()
            .cloned()
            .enumerate()
            .map(|(x, y)| (y, x))
            .collect();

        let mut edge_set: HashSet<Edge> = HashSet::new();
        for e in self.tris.keys().cloned() {
            if !edge_set.contains(&e.rev()) {
                edge_set.insert(e);
            }
        }

        let edges: Vec<(usize, usize)> = edge_set
            .drain()
            .filter(|e| !e.is_ghost())
            .map(|e| {
                (
                    idx_map.get(&e.0).cloned().unwrap(),
                    idx_map.get(&e.1).cloned().unwrap(),
                )
            })
            .collect();

        let mut vis: Visualizer = nodes.into();
        vis.set_edges(edges);
        vis
    }
}
