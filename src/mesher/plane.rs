use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::Instant;

use crate::element::{ElementAssemblage, ElementDescriptor, NodeId};
use crate::spatial::predicates::{self};
use crate::visual::Visualizer;

use super::bounds::{PlaneBoundary, Segment};
use super::{MeshError, Vertex};

use spacemath::two::{dist::Dist, Point};
use spacemath::Orient;

// structs for ordered pairs / triplets of vertex indices
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Triangle(Vertex, Vertex, Vertex);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Edge(Vertex, Vertex);

impl Triangle {
    fn get(self) -> (Point, Point, Point) {
        (self.0.get(), self.1.get(), self.2.get())
    }

    fn perturbed(self) -> (Point, Point, Point) {
        (self.0.perturbed(), self.1.perturbed(), self.2.perturbed())
    }

    fn edges(self) -> [Edge; 3] {
        [
            Edge(self.0, self.1),
            Edge(self.1, self.2),
            Edge(self.2, self.0),
        ]
    }

    fn rot(self) -> Self {
        Triangle(self.1, self.2, self.0)
    }

    fn rev(self) -> Self {
        Triangle(self.2, self.1, self.0)
    }

    fn all(self) -> impl Iterator<Item = Self> {
        vec![
            self,
            self.rot(),
            self.rot().rot(),
            self.rev(),
            self.rev().rot(),
            self.rev().rot().rot(),
        ]
        .into_iter()
    }
}

impl From<(Vertex, Vertex, Vertex)> for Triangle {
    fn from(t: (Vertex, Vertex, Vertex)) -> Triangle {
        Triangle(t.0, t.1, t.2)
    }
}

impl Edge {
    fn rev(self) -> Edge {
        Edge(self.1, self.0)
    }
}

impl From<(Vertex, Vertex)> for Edge {
    fn from(e: (Vertex, Vertex)) -> Edge {
        Edge(e.0, e.1)
    }
}

impl From<Segment> for Edge {
    fn from(s: Segment) -> Edge {
        Edge(s.0, s.1)
    }
}

impl From<Edge> for Segment {
    fn from(e: Edge) -> Segment {
        Segment(e.0, e.1) // TODO, clear up this namespace and consolidate
    }
}

#[derive(Debug)]
pub struct PlaneTriangulation {
    // mesh bounds structure
    pub bound: PlaneBoundary,

    vertices: Vec<Vertex>,

    // map with three entries per triangle for fast lookup
    tris: HashMap<Edge, Vertex>,

    // map storing one copy of each triangle
    // plus the triangle's circumradius for fast lookup
    full_tris: HashMap<Triangle, f64>,

    // record of coappearing vertices in recently added triangles
    recents: HashMap<Vertex, Vertex>,
}

impl PlaneTriangulation {
    pub fn new(bound: PlaneBoundary) -> Self {
        // initialize a new empty triangulation with the given boundary
        PlaneTriangulation {
            bound,
            vertices: Vec::new(),
            tris: HashMap::new(),
            full_tris: HashMap::new(),
            recents: HashMap::new(),
        }
    }

    fn store_vertex<T: Into<Point>>(&mut self, p: T) -> Vertex {
        let res = Vertex::new(p.into());
        self.vertices.push(res);
        res
    }

    fn all_vertices(&self) -> impl Iterator<Item = &'_ Vertex> {
        self.vertices.iter().chain(self.bound.all_vertices())
    }

    fn in_circle<T: Into<Triangle>>(&self, tri: T, x: Vertex, perturb: bool) -> bool {
        // determine whether x lies in the oriented triangle tri's circumcircle
        // perturb determines whether or not to perturb the inputs (which helps avoid chew edge cases)
        let (x, tri) = if perturb {
            (x.perturbed(), tri.into().perturbed())
        } else {
            (x.get(), tri.into().get())
        };

        let x = x.into();
        let (a, b, c) = tri;
        let abc = (a.into(), b.into(), c.into()); // conversion spacemath->spatial TODO remove this

        predicates::in_circle(x, abc)
    }

    fn triangle_dir<T: Into<Triangle>>(&self, tri: T) -> Orient {
        // determine the orientation of a triangle
        // TODO pull out of this method
        let tri: spacemath::two::Triangle = tri.into().get().into();
        tri.dir()
    }

    fn to_left<E: Into<Edge>>(&self, e: E, p: Vertex) -> bool {
        // determine whether p is to the left of ('in front of') e
        // in constrast to triangle_dir, this properly handles ghost points
        // TODO make sure this is in fact the right ghost handling
        let e = e.into();
        self.triangle_dir((e.0, e.1, p)) == Orient::Positive
    }

    fn add_triangle<T: Into<Triangle>>(&mut self, tri: T) -> Result<(), MeshError> {
        // add a triangle to the triangulation
        // only one of the vertices may be a ghost vertex
        // if no vertices are ghosts, as a sanity check ensure the triangle is positive
        let tri = tri.into();
        if self.triangle_dir(tri) != Orient::Positive {
            return Err(MeshError::triangle("negative triangle addition requested"));
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

        let tri_temp: spacemath::two::Triangle = tri.get().into(); // TODO clean up this path
        let cr = tri_temp.circumradius();
        self.full_tris.insert(tri, cr);

        // add to the recent co-appearing points map for later adjacent_one use
        for e in edges.iter().cloned() {
            self.recents.insert(e.0, e.1);
            self.recents.insert(e.1, e.0);
        }

        Ok(())
    }

    fn delete_triangle<T: Into<Triangle>>(&mut self, tri: T) -> Result<(), MeshError> {
        // delete a triangle
        let tri = tri.into();
        let edges = tri.edges();
        if self.tris.get(&edges[0]) == Some(&tri.2) {
            // remove the triangle from both storage structs
            for e in edges.iter() {
                self.tris.remove(e);
            }

            for t in tri.all() {
                if self.full_tris.remove(&t).is_some() {
                    // only try removing until the triangle was successfully removed
                    break;
                }
            }

            Ok(())
        } else {
            Err(MeshError::triangle("triangle to be deleted does not exist"))
        }
    }

    fn adjacent<E: Into<Edge>>(&self, e: E) -> Option<Vertex> {
        // return Some(w) if the positively oriented uvw exists
        self.tris.get(&e.into()).cloned()
    }

    #[allow(dead_code)]
    fn adjacent_one(&self, u: Vertex) -> Option<Edge> {
        // return an arbitrary triangle including u, if one exists
        // if u has been part of a recent triangle, return it
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
        for v in self.all_vertices().cloned() {
            let e = Edge(u, v);
            if let Some(w) = self.tris.get(&e).cloned() {
                return Some((v, w).into());
            }
            if let Some(w) = self.tris.get(&e.rev()).cloned() {
                return Some((w, v).into());
            }
        }

        None
    }

    fn circumcenter<T: Into<Triangle>>(&self, tri: T) -> Option<Point> {
        // find the circumcenter of a given triangle
        // if the triangle is a ghost or degenerate, returns None
        let tri = tri.into().get();
        predicates::circumcenter((tri.0.into(), tri.1.into(), tri.2.into())).map(Into::into)
    }

    fn bowyer_watson_dig(&mut self, u: Vertex, v: Vertex, w: Vertex) {
        // u is a new vertex

        // if wv or vw is a segment, do not cross it; add uvw
        // TODO not super efficient check, used to have info on whether v or w was in the bound
        // now vertices are no longer an enum - worth reimplementing?
        if self.bound.is_segment((w, v)) || self.bound.is_segment((v, w)) {
            self.add_triangle((u, v, w))
                .expect("could not add triangle");
            return;
        }

        // check if uvw is constrained delaunay
        if let Some(x) = self.adjacent((w, v)) {
            if self.triangle_dir((u, v, w)) == Orient::Negative {
                // if u is past vw, uvw isn't delaunay - dig further
                self.delete_triangle((w, v, x))
                    .expect("unreachable - adjacent always returns valid triangles");
                self.bowyer_watson_dig(u, v, x);
                self.bowyer_watson_dig(u, x, w);
                return;
            }

            if self.in_circle((u, v, w), x, false) {
                self.delete_triangle((w, v, x))
                    .expect("unreachable - adjacent always returns valid triangles");
                self.bowyer_watson_dig(u, v, x);
                self.bowyer_watson_dig(u, x, w);
            } else {
                self.add_triangle((u, v, w))
                    .expect("could not add triangle");
            }
        } else {
            self.add_triangle((u, v, w))
                .expect("could not add triangle");
        }
    }

    fn bowyer_watson_insert<T: Into<Triangle>>(&mut self, u: Vertex, tri: T) {
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

        // gift wrapping almost never crosses segments, so it's much cheaper to run the
        // search without visibility checks and just run again if a segment is crossed
        let mut tri: Option<Triangle> = None;
        for v in self.all_vertices().copied() {
            // proceed if v is in front of e and tri is either None or encircling
            if self.to_left(e, v) && tri.map_or(true, |t| self.in_circle(t, v, true)) {
                tri = Some((e.0, e.1, v).into());
            }
        }

        let candidate_node = tri?.2; // return if no triangle was found

        // check if the candidate node is occluded; if so rerun with visibility checks
        if !self.bound.midpoint_visible(e.into(), candidate_node) {
            // run the whole thing again, but with visibility checks
            tri = None;
            for v in self.all_vertices().copied() {
                // proceed if v is in front of e and tri is either None or encircling
                if self.to_left(e, v) && tri.map_or(true, |t| self.in_circle(t, v, true)) {
                    if self.bound.midpoint_visible(e.into(), v) {
                        tri = Some((e.0, e.1, v).into());
                    }
                }
            }
        }

        // return the triangle and the two new directed edges
        tri.map(|t| (t, (t.1, t.2).into(), (t.2, t.0).into()))
    }

    pub fn gift_wrap(&mut self) {
        // gift-wrap the triangulation from scratch
        self.tris.clear();
        self.recents.clear();

        let mut to_finish: HashSet<Edge> = self.bound.all_walls().map(Into::into).collect();

        while !to_finish.is_empty() {
            // remove an element from the set
            let e = to_finish.iter().next().cloned().unwrap();
            to_finish.remove(&e);

            if let Some((tri, a, b)) = self.gift_wrap_finish(e) {
                if let Err(e) = self.add_triangle(tri) {
                    let mut vis = self.visualize();
                    let tp = tri.get();
                    vis.add_points(vec![tp.0.into(), tp.1.into(), tp.2.into()], 1); // TODO switch visual to spacemath

                    vis.draw("err_state.png", ());
                    panic!("error gift-wrapping boundary: {}", e);
                }
                if !to_finish.remove(&a) {
                    // if a was not in set, add its reverse
                    to_finish.insert(a.rev());
                }
                if !to_finish.remove(&b) {
                    to_finish.insert(b.rev());
                }
            }
        }
    }

    fn large_triangle(&self, h: f64) -> Option<Triangle> {
        self.full_tris
            .iter()
            .find(|&(_, &cr)| cr > h)
            .map(|(&t, _)| t)
    }

    fn skinny_triangle(&self, b: f64, h: f64) -> Option<Triangle> {
        // finds both skinny triangles (wrt b) and large triangles (wrt h)
        self.full_tris
            .iter()
            .find(|&(t, &cr)| {
                let (u, v, w) = t.get();
                let min_leg = u.dist(v).min(v.dist(w)).min(w.dist(u));
                ((cr / min_leg) > b) || (cr > h)
            })
            .map(|(&t, _)| t)
    }

    pub fn chew_mesh(&mut self, h: f64) {
        // starting from a rough boundary
        // refine first boundary then interior via chew's 1st method
        eprintln!("beginning chew mesh generation ...");

        // refine the boundary and construct initial triangulation
        eprintln!("finding initial triangulation ...");
        self.bound.divide_all_segments(h);
        self.gift_wrap();
        eprintln!("boundary triangulated\ninserting nodes ...");

        while let Some(tri) = self.large_triangle(h) {
            // tri is a triangle with a too-high circumradius
            // insert its circumcenter
            let center = self.circumcenter(tri).unwrap();
            let center_id = self.store_vertex(center);

            self.bowyer_watson_insert(center_id, tri);
        }
        eprintln!("chew mesh generation complete");
    }

    // maybe move some of these methods to bounds
    fn ruppert_encroaches(&self, p: Vertex, s: Segment) -> bool {
        // check if s is encroached by p by the rules of ruppert meshing
        // putting the visibility check (expensive) after the distance check (cheap)
        // made this way faster, if still too slow implement better visibility check

        if s.has(p) {
            return false; // a segment is not encroached by its own end points
        }

        // check if p is in the diametral circle
        let (a, b) = (s.0.get(), s.1.get());
        let mid = a.mid(b);
        let rad = mid.dist(a);

        if mid.dist(p.get()) > rad {
            return false;
        }

        self.bound.midpoint_visible(s, p)
    }

    fn ruppert_split_segment(&mut self, s: Segment) -> (Segment, Segment, Vertex) {
        // split a bounding segment, using the bowyer-watson algorithm to maintain the delauney property
        // return the child segments and the newly created vertex

        let x = self
            .adjacent(s)
            .expect("this should exist if self was triangulated");
        let Segment(v, w) = s;
        self.delete_triangle((v, w, x)).unwrap();

        let (child_a, child_b, new_v) = self.bound.split_segment(s);
        let u = child_a.1;

        // now it's just like the bowyer_watson insert, but we don't dig through the segment we split
        self.bowyer_watson_dig(u, w, x);
        self.bowyer_watson_dig(u, x, v);

        (child_a, child_b, new_v)
    }

    fn ruppert_split_many(
        &mut self,
        p: Vertex,
        mut to_split: Vec<Segment>,
        recursive: bool,
    ) -> Vec<Vertex> {
        // split the provided segments while their children are still encroached
        // if recursive flag is set, split the children too
        // return the newly generated vertices
        let mut res = Vec::new();

        while let Some(s) = to_split.pop() {
            let (child_a, child_b, new_v) = self.ruppert_split_segment(s);

            if recursive {
                if self.ruppert_encroaches(p, child_a) {
                    to_split.push(child_a);
                }
                if self.ruppert_encroaches(p, child_b) {
                    to_split.push(child_b);
                }
            }

            res.push(new_v);
        }

        res
    }

    fn segments_encroached_by(&self, p: Vertex) -> Option<Vec<Segment>> {
        // find segments which would be encroached by the addition of a trial point
        // returns None if the vector would otherwise be empty
        let segs: Vec<Segment> = self
            .bound
            .all_walls()
            .filter(|&s| self.ruppert_encroaches(p, s))
            .collect();
        if segs.is_empty() {
            None
        } else {
            Some(segs)
        }
    }

    pub fn ruppert_mesh(&mut self, max_size: f64) {
        eprintln!("beginning ruppert mesh generation ...");
        eprint!("triangulating initial boundary ... ");
        let ti = Instant::now();
        self.gift_wrap();
        eprintln!("done in {}ms", ti.elapsed().as_millis());

        // first, split all the walls that are already encroached by boundary vertices
        eprint!("splitting walls ... ");
        let ti = Instant::now();
        let mut to_check: Vec<Vertex> = self.bound.all_vertices().copied().collect();
        while let Some(v) = to_check.pop() {
            if let Some(segs) = self.segments_encroached_by(v) {
                let new_vs = self.ruppert_split_many(v, segs, true);
                to_check.extend(new_vs);
            }
        }
        eprintln!("done in {}ms", ti.elapsed().as_millis());

        eprint!("refining mesh ... ");
        let ti = Instant::now();
        // now, perform the ruppert refinement steps
        let b = 1.415; // minimum skinny triangle ratio for proof of termination
        while let Some(tri) = self.skinny_triangle(b, max_size) {
            let center = self.circumcenter(tri).unwrap(); // todo get rid of wrapping method
            let center = Vertex::new(center); // ok to do this outside of storage

            if let Some(segs) = self.segments_encroached_by(center) {
                self.ruppert_split_many(center, segs, false);
            } else {
                let center_id = self.store_vertex(center.get()); // this'll spin a new id, but that doesn't matter
                self.bowyer_watson_insert(center_id, tri);
            }
        }
        eprintln!("done in {}ms", ti.elapsed().as_millis());
    }

    pub fn visualize(&self) -> Visualizer {
        let nodes: Vec<Point> = self.all_vertices().map(|x| x.get()).collect();

        let idx_map: HashMap<Vertex, usize> = self
            .all_vertices()
            .copied()
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

    pub fn assemble(self) -> Result<ElementAssemblage, &'static str> {
        let mat = self.bound.material().ok_or("missing material definition")?;

        let mut res = ElementAssemblage::new(mat, self.bound.condition().unwrap());

        // build vertex list and translation to elas node ids
        let vertex_list: Vec<(f64, f64)>;
        let vertex_lookup: HashMap<Vertex, NodeId>;

        vertex_list = self.all_vertices().map(|v| v.get().into()).collect();
        let node_ids = res.add_nodes(&vertex_list);
        vertex_lookup = self.all_vertices().copied().zip(node_ids).collect();

        // add all the triangles

        for tri in self.full_tris.into_keys() {
            let tri_nids = [
                *vertex_lookup.get(&tri.0).unwrap(),
                *vertex_lookup.get(&tri.1).unwrap(),
                *vertex_lookup.get(&tri.2).unwrap(),
            ];

            let desc = ElementDescriptor::new(tri_nids);
            res.add_element(desc);
        }

        // pull the various boundary conditions through from bound

        for (id, con) in self.bound.all_constraints() {
            // upgrade the bounds::Vertex to a Vertex
            let id: Vertex = id.into();
            res.add_constraint(*vertex_lookup.get(&id).unwrap(), con);
        }

        for (seg, force) in self.bound.all_distributed_forces() {
            let edg: Edge = seg.into();

            // break out the two end point indices
            let a = *vertex_lookup.get(&edg.0).unwrap();
            let b = *vertex_lookup.get(&edg.1).unwrap();

            res.add_dist_line_force(a, b, force.into());
        }

        Ok(res)
    }
}
