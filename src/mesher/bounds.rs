use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

use crate::spatial::{predicates, Point};
use crate::visual::Visualizer;

use crate::element::loading::Constraint;
use crate::element::material::Material;

// a bounds vertex id
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VId {
    Real(usize),
}

// a directed boundary segment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Segment(pub VId, pub VId);

impl From<(VId, VId)> for Segment {
    fn from(s: (VId, VId)) -> Segment {
        Segment(s.0, s.1)
    }
}

impl Segment {
    fn rev(self) -> Segment {
        Segment(self.1, self.0)
    }

    fn has(self, v: VId) -> bool {
        self.0 == v || self.1 == v
    }
}

// Wall is an iterator over connecting segments
#[derive(Clone, Debug)]
pub struct Wall<'a> {
    seg_map: &'a HashMap<VId, VId>,
    start: VId,
    prev: VId,
    done: bool,
}

impl<'a> Wall<'a> {
    fn new(seg_map: &'a HashMap<VId, VId>, start: VId) -> Self {
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
    seg_map: &'a HashMap<VId, VId>,
    wall_starts: &'a [VId],
    wall: Wall<'a>,
    wall_idx: usize,
}

impl<'a> AllWalls<'a> {
    fn new(seg_map: &'a HashMap<VId, VId>, wall_starts: &'a [VId]) -> Self {
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
    vertices: Vec<(f64, f64)>,
    wall_starts: Vec<VId>,
    seg_map: HashMap<VId, VId>,
    seg_set: HashSet<Segment>,

    // storage for undivided walls (for cheaper visibility tests)
    base_wall_starts: Vec<VId>,
    base_seg_map: HashMap<VId, VId>,

    // also store some BC / material information
    thickness: Option<f64>,
    material: Option<Material>,
    constraints: HashMap<VId, Constraint>,
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

            thickness: None,
            material: None,
            constraints: HashMap::new(),
            distributed_forces: HashMap::new(),
            distributed_constraints: HashMap::new(),
        }
    }

    pub fn store_vertex<T: TryInto<(f64, f64)>>(&mut self, p: T) -> VId {
        if let Ok((x, y)) = p.try_into() {
            self.vertices.push((x, y));
            VId::Real(self.vertices.len() - 1)
        } else {
            panic!("bad vertex input");
        }
    }

    pub fn store_segment<S: Into<Segment>>(&mut self, s: S) {
        let s = s.into();

        let a = self.get(s.0).expect("bad index");
        let b = self.get(s.1).expect("bad index");

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

        let a = self.get(s.0).expect("bad index");
        let b = self.get(s.1).expect("bad index");

        // possibly avoid this O(n) check for at-scale work
        if self.intersects(a, b, Some(s)) {
            panic!("segment overlaps with existing segment");
        }

        self.seg_map.insert(s.0, s.1);
        self.base_seg_map.insert(s.0, s.1);
        self.seg_set.insert(s);
    }

    pub fn set_thickness(&mut self, t: f64) {
        self.thickness = Some(t);
    }

    pub fn thickness(&self) -> Option<f64> {
        self.thickness
    }

    pub fn set_material(&mut self, m: Material) {
        self.material = Some(m);
    }

    pub fn material(&self) -> Option<Material> {
        self.material
    }

    pub fn store_constraint(&mut self, v: VId, c: Constraint) {
        self.constraints.insert(v, c);
    }

    pub fn store_distributed_force(&mut self, v: VId, w: VId, f: Point) {
        self.distributed_forces.insert((v, w).into(), f);
    }

    pub fn store_distributed_constraint(&mut self, v: VId, w: VId, c: Constraint) {
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
    }

    fn store_segment_unchecked<S: Into<Segment>>(&mut self, s: S) {
        // for use when non-overlap is guaranteed
        // e.g. for splitting existing segments
        let s = s.into();

        self.seg_map.insert(s.0, s.1);
    }

    pub fn get(&self, id: VId) -> Option<(f64, f64)> {
        match id {
            VId::Real(i) => self.vertices.get(i).cloned(),
        }
    }

    pub fn all_vid(&self) -> impl Iterator<Item = VId> {
        (0..self.vertices.len()).map(VId::Real)
    }

    pub fn get_segment_points(&self, seg: Segment) -> Option<(Point, Point)> {
        let Segment(u, v) = seg;
        Some((self.get(u)?.into(), self.get(v)?.into()))
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

            let a = self.get(s.0).unwrap();
            let b = self.get(s.1).unwrap();
            if predicates::segments_intersect((a.into(), b.into()), (p.into(), q.into())) {
                return true;
            }
        }
        false
    }

    pub fn store_polygon<T: IntoIterator<Item = (f64, f64)>>(&mut self, poly: T) -> Vec<VId> {
        // store an ordered set of points as a polygon
        // the polygon should be oriented so positive triangles off the edges are inside the body
        // TODO enforce this
        let mut poly = poly.into_iter();
        let mut ids = Vec::new();
        let start = poly.next().unwrap();
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
        let a: Point = self.get(s.0).expect("bad index").into();
        let b: Point = self.get(s.1).expect("bad index").into();

        let dist = a.dist(b);
        let lower_bound = h;
        let upper_bound = h * (3.0_f64).sqrt();

        // find number of segments to split into
        let lower_num = dist / upper_bound;
        let upper_num = dist / lower_bound;

        if lower_num.floor() == upper_num.floor() {
            panic!("cannot find valid segment division");
        }

        // define new segment interval
        let num = upper_num.floor();
        let segment_length = dist / num;
        let leg = (b - a).unit() * segment_length;

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

    pub fn all_constraints(&self) -> Vec<(VId, Constraint)> {
        self.constraints.iter().map(|(&v, &c)| (v, c)).collect()
    }

    pub fn visualize(&self) -> Visualizer {
        let nodes: Vec<Point> = self
            .all_vid()
            .map(|x| self.get(x).unwrap().into())
            .collect();

        let mut edges = Vec::new();
        for s in self.seg_set.iter() {
            match s {
                Segment(VId::Real(i), VId::Real(j)) => edges.push((*i, *j)),
            }
        }

        let mut vis: Visualizer = nodes.into();
        vis.set_edges(edges);
        vis
    }
}
