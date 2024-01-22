use std::collections::{HashMap, HashSet};

use crate::visual::Visualizer;

use crate::element::constraint::Constraint;
use crate::element::material::Material;
use crate::element::strain::Condition;

use super::Vertex;

use spacemath::two::dist::Dist;
use spacemath::two::intersect::Intersect;
use spacemath::two::{Circle, Point};

// a directed boundary segment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Segment(pub Vertex, pub Vertex);

impl From<(Vertex, Vertex)> for Segment {
    fn from(s: (Vertex, Vertex)) -> Segment {
        Segment(s.0, s.1)
    }
}

impl Segment {
    pub fn rev(self) -> Segment {
        Segment(self.1, self.0)
    }

    pub fn has(self, v: Vertex) -> bool {
        self.0 == v || self.1 == v
    }

    pub fn get(self) -> (Point, Point) {
        (self.0.get(), self.1.get())
    }

    pub fn mid(self) -> Point {
        self.0.get().mid(self.1.get())
    }

    pub fn len(self) -> f64 {
        self.0.get().dist(self.1.get())
    }

    pub fn diametral(self) -> Circle {
        Circle::new(self.mid(), self.len() / 2.0)
    }
}

// Wall is an iterator over connecting segments
#[derive(Clone, Debug)]
pub struct Wall<'a> {
    seg_map: &'a HashMap<Vertex, Vertex>,
    start: Vertex,
    prev: Vertex,
    done: bool,
}

impl<'a> Wall<'a> {
    fn new(seg_map: &'a HashMap<Vertex, Vertex>, start: Vertex) -> Self {
        Self {
            seg_map,
            start,
            prev: start,
            done: false,
        }
    }
}

impl<'a> Iterator for Wall<'a> {
    type Item = Segment;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if let Some(cur) = self.seg_map.get(&self.prev).cloned() {
            if cur == self.start {
                self.done = true;
            }

            let seg = (self.prev, cur).into();
            self.prev = cur;
            Some(seg)
        } else {
            self.done = true;
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct AllWalls<'a> {
    seg_map: &'a HashMap<Vertex, Vertex>,
    wall_starts: &'a [Vertex],
    wall: Wall<'a>,
    wall_idx: usize,
}

impl<'a> AllWalls<'a> {
    fn new(seg_map: &'a HashMap<Vertex, Vertex>, wall_starts: &'a [Vertex]) -> Self {
        assert!(
            !wall_starts.is_empty(),
            "cannot iterate over empty wall list"
        );

        Self {
            seg_map,
            wall_starts,
            wall: Wall::new(seg_map, wall_starts[0]),
            wall_idx: 0,
        }
    }
}

impl<'a> Iterator for AllWalls<'a> {
    type Item = Segment;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(seg) = self.wall.next() {
            Some(seg)
        } else {
            self.wall_idx += 1;
            if self.wall_idx >= self.wall_starts.len() {
                None
            } else {
                self.wall = Wall::new(self.seg_map, self.wall_starts[self.wall_idx]);
                Some(self.wall.next().expect("zero length wall"))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlaneBoundary {
    vertices: Vec<Vertex>,

    wall_starts: Vec<Vertex>,
    seg_map: HashMap<Vertex, Vertex>,
    seg_set: HashSet<Segment>,

    // storage for undivided walls (for cheaper visibility tests)
    base_wall_starts: Vec<Vertex>,
    base_seg_map: HashMap<Vertex, Vertex>,

    // in_diametral_map: HashMap<Segment, >

    // also store some BC / material information
    condition: Option<Condition>,
    material: Option<Material>,
    constraints: HashMap<Vertex, Constraint>,
    distributed_forces: HashMap<Segment, Point>,
    distributed_constraints: HashMap<Segment, Constraint>,
}

impl PlaneBoundary {
    pub fn new() -> Self {
        PlaneBoundary {
            vertices: Vec::new(),

            wall_starts: Vec::new(),
            seg_map: HashMap::new(),
            seg_set: HashSet::new(),

            base_wall_starts: Vec::new(),
            base_seg_map: HashMap::new(),

            condition: None,
            material: None,
            constraints: HashMap::new(),
            distributed_forces: HashMap::new(),
            distributed_constraints: HashMap::new(),
        }
    }

    pub fn store_vertex<T: Into<Point>>(&mut self, p: T) -> Vertex {
        let res = Vertex::new(p.into());
        self.vertices.push(res);
        res
    }

    pub fn store_segment<S: Into<Segment>>(&mut self, s: S) {
        let s = s.into();

        let (a, b) = s.get();

        // possibly avoid this O(n) check for at-scale work
        if self.intersects(a, b, Some(s)) {
            panic!("segment overlaps with existing segment");
        }

        self.seg_map.insert(s.0, s.1);
        self.seg_set.insert(s);
    }

    fn store_base_segment<S: Into<Segment>>(&mut self, s: S) {
        // store a segment into both the main and base maps
        let s = s.into();

        let (a, b) = s.get();

        // possibly avoid this O(n) check for at-scale work
        if self.intersects(a, b, Some(s)) {
            panic!("segment overlaps with existing segment");
        }

        self.seg_map.insert(s.0, s.1);
        self.base_seg_map.insert(s.0, s.1);
        self.seg_set.insert(s);
    }

    pub fn set_condition(&mut self, c: Condition) {
        self.condition = Some(c);
    }

    pub fn condition(&self) -> Option<Condition> {
        self.condition
    }

    pub fn set_material(&mut self, m: Material) {
        self.material = Some(m);
    }

    pub fn material(&self) -> Option<Material> {
        self.material
    }

    pub fn store_constraint(&mut self, v: Vertex, c: Constraint) {
        self.constraints.insert(v, c);
    }

    pub fn store_distributed_force(&mut self, v: Vertex, w: Vertex, f: Point) {
        self.distributed_forces.insert((v, w).into(), f);
    }

    pub fn store_distributed_constraint(&mut self, v: Vertex, w: Vertex, c: Constraint) {
        self.distributed_constraints.insert((v, w).into(), c);
    }

    pub fn is_segment<S: Into<Segment>>(&self, s: S) -> bool {
        let s = s.into();
        self.seg_set.contains(&s)
    }

    fn remove_segment<S: Into<Segment>>(&mut self, s: S) {
        let s = s.into();

        self.seg_map.remove(&s.0);
        self.seg_set.remove(&s);

        // should probably also remove any associated forces / constraints
        self.distributed_forces.remove(&s);
        self.distributed_constraints.remove(&s);
    }

    fn store_segment_unchecked<S: Into<Segment>>(&mut self, s: S) {
        // for use when non-overlap is guaranteed
        // e.g. for splitting existing segments
        let s = s.into();

        self.seg_map.insert(s.0, s.1);
        self.seg_set.insert(s);
    }

    pub fn all_vertices(&self) -> impl Iterator<Item = &'_ Vertex> {
        self.vertices.iter()
    }

    fn intersects<T: Into<Point> + Copy>(&self, p: T, q: T, ignore: Option<Segment>) -> bool {
        // determine whether the segment pq intersects an existing segment
        // if ignore is Some, ignore intersection with segments sharing vertices
        for s in self.seg_set.iter() {
            if let Some(ignore_seg) = ignore {
                if s.has(ignore_seg.0) || s.has(ignore_seg.1) {
                    continue;
                }
            }

            let (a, b) = s.get();

            // TODO find a better way to deal with namespace
            // TODO further integrate spacemath into function signatures
            let ab = spacemath::two::Segment::new(a, b);
            let pq = spacemath::two::Segment::new(p.into(), q.into());
            if ab.intersects(&pq).is_nonzero() {
                return true;
            }
        }
        false
    }

    pub fn midpoint_visible(&self, s: Segment, x: Vertex) -> bool {
        // determine whether the midpoint of s is visible from v
        const TOLERANCE: f64 = 1e-6;

        let (p, q) = s.get();
        let m = p.mid(q); // midpoint

        // check if any wall obstructs mx
        // make sure to skip walls which m or x contact
        // this was previously done by checking if e.0 or e.1 are wall endpoints
        // now that only base walls are checked, a spatial predicate is used
        for wall in self.all_base_walls() {
            let (a, b) = wall.get();
            let x = x.get();
            let ab = spacemath::two::Segment::new(a, b);
            let mx = spacemath::two::Segment::new(m, x);
            if ab.intersects(&mx).is_nonzero() {
                if !((ab.dist(m) < TOLERANCE) || (ab.dist(x) < TOLERANCE)) {
                    // if they intersect and neither m nor x lie on ab
                    return false;
                }
            }
        }

        true
    }

    pub fn store_polygon<T>(&mut self, poly: &[T]) -> Vec<Vertex>
    where
        T: Into<Point> + Copy,
    {
        // store an ordered set of points as a polygon
        // the polygon should be oriented so positive triangles off the edges are inside the body

        let mut poly = poly.into_iter().copied().map(|p| p.into());
        let mut ids = Vec::new();
        let start: Point = poly.next().unwrap();
        let start_idx = self.store_vertex(start);
        ids.push(start_idx);

        self.wall_starts.push(start_idx);
        self.base_wall_starts.push(start_idx);

        let mut prev_idx = start_idx;
        for p in poly {
            let p_idx = self.store_vertex(p);
            ids.push(p_idx);
            self.store_base_segment((prev_idx, p_idx));

            prev_idx = p_idx;
        }

        self.store_base_segment((prev_idx, start_idx));
        ids
    }

    pub fn divide_segment(&mut self, s: Segment, h: f64) {
        // divide a segment into chunks with length on the interval [h, sqrt(3)*h]
        assert!(self.is_segment(s));
        let (a, b) = s.get();

        let dist = a.dist(b);
        let lower_bound = h;
        let upper_bound = h * (3.0_f64).sqrt();

        // find number of segments to split into
        let lower_num = (dist / upper_bound).floor();
        let upper_num = (dist / lower_bound).floor();

        if lower_num == upper_num {
            panic!(
                "cannot find valid segment division for segment of length {}",
                dist
            );
        }

        // define new segment interval
        let num = upper_num;
        let segment_length = dist / num;
        let leg = (b - a).to_unit() * segment_length;

        // find whether there are distributed forces or constraints on the segment
        let dist_f = self
            .distributed_forces
            .remove(&s)
            .or_else(|| self.distributed_forces.remove(&s.rev()));

        let dist_c = self
            .distributed_constraints
            .remove(&s)
            .or_else(|| self.distributed_constraints.remove(&s.rev()));

        self.remove_segment(s);
        let start = s.0;
        let end = s.1;
        let mut cur = start;
        let mut new_point = a + leg;

        // move along, adding new points and segments
        for _ in 0..((num as i64) - 1) {
            let new_point_id = self.store_vertex(new_point);
            self.store_segment_unchecked((cur, new_point_id));

            // if there are distributed attributes, insert them
            if let Some(f) = dist_f {
                self.distributed_forces
                    .insert((cur, new_point_id).into(), f);
            }

            if let Some(c) = dist_c {
                self.distributed_constraints
                    .insert((cur, new_point_id).into(), c);

                // distributed constraints are just bookkeeping
                // they mark segments where new points should have cosntraints
                self.constraints.insert(new_point_id, c);
            }

            cur = new_point_id;
            new_point = new_point + leg;
        }

        if let Some(c) = dist_c {
            self.constraints.insert(start, c);
            self.constraints.insert(end, c);
        }

        self.store_segment_unchecked((cur, end));

        if let Some(f) = dist_f {
            self.distributed_forces.insert((cur, end).into(), f);
        }
    }

    pub fn split_segment(&mut self, s: Segment) -> (Segment, Segment, Vertex) {
        // split a segment in two, storing the result and also returning it for later use
        assert!(self.is_segment(s));
        let Segment(a, b) = s; // points

        let m = a.get().mid(b.get());
        let m = self.store_vertex(m);

        // find whether there are distributed forces or constraints on the segment
        let dist_f = self
            .distributed_forces
            .remove(&s)
            .or_else(|| self.distributed_forces.remove(&s.rev()));

        let dist_c = self
            .distributed_constraints
            .remove(&s)
            .or_else(|| self.distributed_constraints.remove(&s.rev()));

        self.remove_segment(s);

        self.store_segment_unchecked((a, m));
        self.store_segment_unchecked((m, b));

        if let Some(c) = dist_c {
            self.distributed_constraints.insert((a, m).into(), c);
            self.distributed_constraints.insert((m, b).into(), c);
        }

        if let Some(f) = dist_f {
            self.distributed_forces.insert((a, m).into(), f);
            self.distributed_forces.insert((m, b).into(), f);
        }

        ((a, m).into(), (m, b).into(), m)
    }

    pub fn divide_all_segments(&mut self, h: f64) {
        for s in self.seg_set.clone().into_iter() {
            // seg_set becomes outdated, but divide_segment should only touch segment s
            self.divide_segment(s, h);
        }
    }

    pub fn all_walls(&self) -> AllWalls<'_> {
        // an iterator over every boundary segment
        AllWalls::new(&self.seg_map, &self.wall_starts)
    }

    pub fn all_base_walls(&self) -> AllWalls<'_> {
        // an iterator over every base boundary segment
        // does not include wall subdivisions, for faster visibility checks
        AllWalls::new(&self.base_seg_map, &self.base_wall_starts)
    }

    pub fn all_distributed_forces(&self) -> Vec<(Segment, Point)> {
        self.distributed_forces
            .iter()
            .map(|(&s, &f)| (s, f))
            .collect()
    }

    pub fn all_constraints(&self) -> Vec<(Vertex, Constraint)> {
        // include the vertices of distributed constraints

        let mut res: HashMap<Vertex, Constraint> = HashMap::new();

        for (v, c) in self.constraints.iter() {
            res.insert(*v, *c);
        }

        for (s, c) in self.distributed_constraints.iter() {
            res.insert(s.0, *c);
            res.insert(s.1, *c); // will just overwrite any duplicates
        }

        res.into_iter().collect()
    }

    pub fn visualize(&self) -> Visualizer {
        let nodes: Vec<Point> = self.all_vertices().map(|x| x.get()).collect();

        // for visualizer plumbing
        let indices: HashMap<Vertex, usize> = self
            .all_vertices()
            .copied()
            .enumerate()
            .map(|(x, y)| (y, x))
            .collect();

        let mut edges = Vec::new();
        for s in self.seg_set.iter() {
            edges.push((indices[&s.0], indices[&s.1]));
        }

        let mut vis: Visualizer = nodes.into();
        vis.set_edges(edges);
        vis
    }
}
