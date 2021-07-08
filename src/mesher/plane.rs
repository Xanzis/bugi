use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::convert::{From, Into, TryInto};
use std::hash::{Hash, Hasher};
use std::iter;

use crate::element::ElementAssemblage;
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

impl From<bounds::VIdx> for VIdx {
    fn from(id: bounds::VIdx) -> Self {
        VIdx::Bound(id)
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
        Edge(s.0.into(), s.1.into())
    }
}

#[derive(Clone, Debug)]
pub struct PlaneTriangulation {
    // mesh bounds structure
    pub bound: PlaneBoundary,

    // vector of vertex locations
    vertices: Vec<(f64, f64)>,

    // map with three entries per triangle for fast lookup
    tris: HashMap<Edge, VIdx>,
    full_tris: HashSet<Triangle>,

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
            full_tris: HashSet::new(),
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

    fn get_perturbed(&self, v: VIdx) -> Option<(f64, f64)> {
        // retrieve a vertex, perturbing it slightly using a hash of the VIdx
        // TODO confirm this is sound
        const MAX_PERTURB: f64 = 1e-9;
        let mut hasher = DefaultHasher::new();
        v.hash(&mut hasher);
        let perturb = (hasher.finish() as f64 / u64::MAX as f64) * MAX_PERTURB;

        self.get(v).map(|(x, y)| (x + perturb, y + perturb))
    }

    fn all_vidx(&self) -> impl Iterator<Item = VIdx> {
        // return an iterator over all valid VIdxs
        (0..self.vertices.len())
            .map(|x| VIdx::Real(x))
            .chain(self.bound.all_vidx().map(|x| VIdx::Bound(x)))
            .chain(iter::once(VIdx::Ghost))
    }

    fn all_triangles<'a>(&'a self) -> impl Iterator<Item = Triangle> + 'a {
        self.full_tris.iter().cloned()
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

    fn get_triangle_points_perturbed(&self, tri: Triangle) -> Option<(Point, Point, Point)> {
        if tri.is_ghost() {
            return None;
        }
        let Triangle(u, v, w) = tri;
        Some((
            self.get_perturbed(u).unwrap().into(),
            self.get_perturbed(v).unwrap().into(),
            self.get_perturbed(w).unwrap().into(),
        ))
    }

    fn in_circle<T: Into<Triangle>>(&self, tri: T, x: VIdx, perturb: bool) -> bool {
        // determine whether x lies in the oriented triangle tri's circumcircle
        // perturb determines whether or not to perturb the inputs (which helps avoid chew edge cases)
        let tri = tri.into();

        // end immediately for doubly-ghost edge cases
        if tri.ghost_count() > 1 {
            return false;
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
            return false;
        }

        // don't check tri for positivity, important that negative triangles return inverse results
        if perturb {
            predicates::in_circle(
                self.get_perturbed(x).expect("nonexistent vidx").into(),
                self.get_triangle_points_perturbed(tri).unwrap(),
            )
        } else {
            predicates::in_circle(
                self.get(x).expect("nonexistent vidx").into(),
                self.get_triangle_points(tri).unwrap(),
            )
        }
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

        self.full_tris.insert(tri);

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
            // remove the triangle from both storage structs
            for e in edges.iter() {
                self.tris.remove(e);
            }

            for t in tri.all() {
                if self.full_tris.remove(&t) {
                    // only try removing until the triangle was successfully removed
                    break;
                }
            }

            Ok(())
        } else {
            Err(MeshError::triangle("triangle to be deleted does not exist"))
        }
    }

    fn disp_vertex(&self, v: VIdx) -> String {
        if let Some(p) = self.get(v) {
            format!("({:1.4}, {:1.4})", p.0, p.1)
        } else {
            "None".to_string()
        }
    }

    fn disp_triangle<T: Into<Triangle>>(&self, tri: T) -> String {
        let tri = tri.into();
        format!(
            "Tri {}, {}, {}",
            self.disp_vertex(tri.0),
            self.disp_vertex(tri.1),
            self.disp_vertex(tri.2)
        )
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

    fn midpoint_visible<E: Into<Edge>>(&self, e: E, v: VIdx) -> bool {
        // determine whether the midpoint of e is visible from v
        let e = e.into();
        if let (Some(p), Some(q), Some(x)) = (self.get(e.0), self.get(e.1), self.get(v)) {
            let p: Point = p.into();
            let m = p.mid(q.into()); // midpoint
            let x: Point = x.into();

            // check if any wall obstructs mx
            for wall in self.bound.all_walls().into_iter() {
                let (a_id, b_id) = (VIdx::Bound(wall.0), VIdx::Bound(wall.1));
                if a_id == e.0
                    || a_id == e.1
                    || a_id == v
                    || b_id == e.0
                    || b_id == e.1
                    || b_id == v
                {
                    // don't check walls which touch the query points (they are always in contact)
                    continue;
                }

                if let Some((a, b)) = self.bound.get_segment_points(wall) {
                    if predicates::segments_intersect((a, b), (m, x)) {
                        return false;
                    }
                }
            }
            true
        } else {
            // default false, TODO make sure this makes sense
            false
        }
    }

    fn circumradius<T: Into<Triangle>>(&self, tri: T) -> Option<f64> {
        // determine the circumradius of the given triangle
        // if triangle is a ghost etc. returns None
        let tri = tri.into();
        self.get_triangle_points(tri).map(predicates::circumradius)
    }

    fn circumcenter<T: Into<Triangle>>(&self, tri: T) -> Option<Point> {
        // find the circumcenter of a given triangle
        // if the triangle is a ghost or degenerate, returns None
        let tri = tri.into();
        if let Some(tri_points) = self.get_triangle_points(tri) {
            predicates::circumcenter(tri_points)
        } else {
            None
        }
    }

    fn bowyer_watson_dig(&mut self, u: VIdx, v: VIdx, w: VIdx) {
        // u is a new vertex

        // if wv or vw is a segment, do not cross it; add uvw
        if let (VIdx::Bound(a), VIdx::Bound(b)) = (w, v) {
            if self.bound.is_segment((a, b)) || self.bound.is_segment((b, a)) {
                self.add_triangle((u, v, w))
                    .expect("could not add triangle");
                return;
            }
        }

        // check if uvw is constrained delaunay
        if let Some(x) = self.adjacent((w, v)) {
            if self.triangle_dir((u, v, w)) == Some(Orient::Negative) {
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

        let mut tri: Option<Triangle> = None;
        for v in self.all_vidx() {
            // do not construct doubly-ghost triangles
            if e.is_ghost() && v.is_ghost() {
                continue;
            }
            // proceed if v is in front of e and tri is either None or encircling
            if self.to_left(e, v) && tri.map_or(true, |t| self.in_circle(t, v, true)) {
                if self.midpoint_visible(e, v) {
                    tri = Some((e.0, e.1, v).into());
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

        let mut to_finish: HashSet<Edge> = HashSet::new();

        for e in self.bound.all_walls().into_iter().map(|s| s.into()) {
            to_finish.insert(e);
        }

        while !to_finish.is_empty() {
            // remove an element from the set
            let e = to_finish.iter().next().cloned().unwrap();
            to_finish.remove(&e);

            if let Some((tri, a, b)) = self.gift_wrap_finish(e) {
                if let Err(e) = self.add_triangle(tri) {
                    let mut vis = self.visualize();
                    if let Some(tp) = self.get_triangle_points(tri) {
                        vis.add_points(vec![tp.0, tp.1, tp.2], 1);
                    }
                    vis.draw(format!("err_state.png").as_str(), ());
                    panic!("error gift-wrapping boundary");
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
        // find a triangle with circumradius larger than h if one exists
        self.all_triangles().find(|&t| {
            if let Some(rad) = self.circumradius(t) {
                rad > h
            } else {
                false
            }
        })
    }

    pub fn chew_mesh(&mut self, h: f64) {
        // starting from a rough boundary
        // refine first boundary then interior via chew's 1st method

        // refine the boundary and construct initial triangulation
        self.bound.divide_all_segments(h);
        self.gift_wrap();

        //while let Some(tri) = self.all_triangles().find(|&t| if let Some(rad) = self.circumradius(t) { rad > h } else { false }) {
        while let Some(tri) = self.large_triangle(h) {
            // tri is a triangle with a too-high circumradius
            // insert its circumcenter
            let center = self
                .circumcenter(tri)
                .expect("unreachable - tri with circumradius always has circumcenter");
            let center_id = self.store_vertex(center);

            self.bowyer_watson_insert(center_id, tri);
        }
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

    pub fn assemble(self) -> Result<ElementAssemblage, &'static str> {
        let mat = self.bound.material().ok_or("missing material definition")?;

        // TODO add logic when 1D/3D is implemented
        let mut res = ElementAssemblage::new(2, mat);

        if let Some(t) = self.bound.thickness() {
            res.set_thickness(t);
        }

        let mut vertex_list: Vec<(f64, f64)> = Vec::new();
        let mut vertex_lookup: HashMap<VIdx, usize> = HashMap::new();

        // translate vertex ids to indices in a now-frozen vertex list

        for id in self.all_vidx() {
            vertex_lookup.insert(id, vertex_list.len());
            vertex_list.push(self.get(id).unwrap());
        }

        res.add_nodes(vertex_list);

        // add all the triangles

        for tri in self.full_tris.into_iter() {
            // TODO avoid these heap allocations by using elas' triangle insertion method
            let tri_def = vec![
                *vertex_lookup.get(&tri.0).unwrap(),
                *vertex_lookup.get(&tri.1).unwrap(),
                *vertex_lookup.get(&tri.2).unwrap(),
            ];

            res.add_element(tri_def);
        }

        // pull the various boundary conditions through from bound

        for (id, con) in self.bound.all_constraints() {
            // upgrade the bounds::VIdx to a VIdx
            let id: VIdx = id.into();
            res.add_constraint(*vertex_lookup.get(&id).unwrap(), con);
        }

        for (seg, force) in self.bound.all_distributed_forces() {
            let edg: Edge = seg.into();

            // break out the two end point indices
            let a = *vertex_lookup.get(&edg.0).unwrap();
            let b = *vertex_lookup.get(&edg.1).unwrap();

            res.add_dist_line_force(a, b, force);
        }

        Ok(res)
    }
}
