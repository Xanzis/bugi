use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

use crate::spatial::{predicates, Point};
use crate::visual::Visualizer;

// a bounds vertex id
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VIdx {
    Real(usize),
}

// a directed boundary segment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Segment(pub VIdx, pub VIdx);

impl From<(VIdx, VIdx)> for Segment {
    fn from(s: (VIdx, VIdx)) -> Segment {
        Segment(s.0, s.1)
    }
}

impl Segment {
    fn rev(self) -> Segment {
        Segment(self.1, self.0)
    }

    fn has(self, v: VIdx) -> bool {
        self.0 == v || self.1 == v
    }
}

#[derive(Clone, Debug)]
pub struct PlaneBoundary {
    vertices: Vec<(f64, f64)>,
    wall_starts: Vec<VIdx>,
    seg_map: HashMap<VIdx, VIdx>,
    seg_set: HashSet<Segment>,
}

impl PlaneBoundary {
    pub fn new() -> Self {
        PlaneBoundary {
            vertices: Vec::new(),
            wall_starts: Vec::new(),
            seg_map: HashMap::new(),
            seg_set: HashSet::new(),
        }
    }

    fn store_vertex<T: TryInto<(f64, f64)>>(&mut self, p: T) -> VIdx {
        if let Ok((x, y)) = p.try_into() {
            self.vertices.push((x, y));
            VIdx::Real(self.vertices.len() - 1)
        } else {
            panic!("bad vertex input");
        }
    }

    fn store_segment<S: Into<Segment>>(&mut self, s: S) {
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

    pub fn get(&self, id: VIdx) -> Option<(f64, f64)> {
        match id {
            VIdx::Real(i) => self.vertices.get(i).cloned(),
        }
    }

    pub fn all_vidx(&self) -> impl Iterator<Item = VIdx> {
        (0..self.vertices.len()).map(|x| VIdx::Real(x))
    }

    fn get_segment_points(&self, seg: Segment) -> Option<(Point, Point)> {
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

    pub fn store_polygon<T: IntoIterator<Item = (f64, f64)>>(&mut self, poly: T) {
        // store an ordered set of points as a polygon
        // the polygon should be oriented so positive triangles off the edges are inside the body
        // TODO enforce this
        let mut poly = poly.into_iter();
        let start = poly.next().unwrap();
        let start_idx = self.store_vertex(start);
        self.wall_starts.push(start_idx);

        let mut prev_idx = start_idx;
        for p in poly {
            let p_idx = self.store_vertex(p);
            self.store_segment((prev_idx, p_idx));

            prev_idx = p_idx;
        }

        self.store_segment((prev_idx, start_idx));
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

        self.remove_segment(s);
        let start = s.0;
        let end = s.1;
        let mut cur = start;
        let mut new_point = a + leg;

        // move along, adding new points and segments
        for _ in 0..((num as i64) - 1) {
            let new_point_idx = self.store_vertex(new_point);
            self.store_segment_unchecked((cur, new_point_idx));

            cur = new_point_idx;
            new_point = new_point + leg;
        }

        self.store_segment_unchecked((cur, end));
    }

    pub fn divide_all_segments(&mut self, h: f64) {
        for s in self.seg_set.clone().into_iter() {
            // seg_set becomes outdated, but divide_segment should only touch segment s
            self.divide_segment(s, h);
        }
    }

    pub fn wall(&self, p: VIdx) -> Vec<Segment> {
        // return a vector of all contiguous walls following p
        // if the walls are a closed loop, finish before repeating a segment
        let mut res: Vec<Segment> = Vec::new();
        let start = p;
        let mut past = p;
        while let Some(current) = self.seg_map.get(&past).cloned() {
            res.push((past, current).into());

            if current == start {
                break;
            }
            past = current;
        }
        res
    }

    pub fn all_walls(&self) -> Vec<Segment> {
        // return a vector of all walls of the boundary, oriented with inside on the left
        self.wall_starts
            .iter()
            .flat_map(|p| self.wall(*p).into_iter())
            .collect()
    }

    pub fn visualize(&self) -> Visualizer {
        let nodes: Vec<Point> = self
            .all_vidx()
            .map(|x| self.get(x).unwrap().into())
            .collect();

        let mut edges = Vec::new();
        for s in self.seg_set.iter() {
            match s {
                Segment(VIdx::Real(i), VIdx::Real(j)) => edges.push((*i, *j)),
            }
        }

        let mut vis: Visualizer = nodes.into();
        vis.set_edges(edges);
        vis
    }
}
