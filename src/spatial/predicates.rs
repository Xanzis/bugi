use std::convert::TryInto;

use crate::matrix::{ConstMatrix, Inverse};

use super::Point;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orient {
    Negative,
    Positive,
    Zero,
}

#[allow(dead_code)]
pub fn tetrahedron_dir(tet: (Point, Point, Point, Point)) -> Orient {
    // return the orientation of a tetrahedron
    let (a, b, c, d) = tet;

    let mat = ConstMatrix::<16>::from_flat_array(
        4,
        [
            a[0], a[1], a[2], 1.0, b[0], b[1], b[2], 1.0, c[0], c[1], c[2], 1.0, d[0], d[1], d[2],
            1.0,
        ],
    );

    match mat.determinant() {
        Ok(val) => {
            if val == 0.0 {
                Orient::Zero
            } else if val > 0.0 {
                Orient::Positive
            } else {
                Orient::Negative
            }
        }
        Err(_) => {
            // error in determinant() implies det = 0
            // TODO handle various error types
            Orient::Zero
        }
    }
}

pub fn bary_coor(p: Point, tri: (Point, Point, Point)) -> (f64, f64, f64) {
    // returns the barymetric coordinates of a point in a triangle
    let (p1, p2, p3) = tri;
    let (x1, y1) = p1.try_into().unwrap();
    let (x2, y2) = p2.try_into().unwrap();
    let (x3, y3) = p3.try_into().unwrap();
    let (x, y) = p.try_into().unwrap();

    let denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    let a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom;
    let b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom;
    let c = 1.0 - a - b;
    (a, b, c)
}

pub fn in_triangle(p: Point, tri: (Point, Point, Point)) -> bool {
    // determine whether p is within the given triangle
    let (a, b, c) = bary_coor(p, tri);
    let rng = 0.0..=1.0;
    rng.contains(&a) && rng.contains(&b) && rng.contains(&c)
}

pub fn in_circle(p: Point, tri: (Point, Point, Point)) -> bool {
    // determine whether p lies within tri's circumcircle
    // tri's points should be in counterclockwise order
    let (a, b, c) = tri;

    // TODO replace with faster / more accurate 3x3 determinant
    let mat = ConstMatrix::<16>::from_flat_array(
        4,
        [
            a[0],
            a[1],
            a[0].powi(2) + a[1].powi(2),
            1.0,
            b[0],
            b[1],
            b[0].powi(2) + b[1].powi(2),
            1.0,
            c[0],
            c[1],
            c[0].powi(2) + c[1].powi(2),
            1.0,
            p[0],
            p[1],
            p[0].powi(2) + p[1].powi(2),
            1.0,
        ],
    );

    if let Ok(val) = mat.determinant() {
        val > 0.0
    } else {
        false
    }
}

#[allow(dead_code)]
pub fn in_sphere(p: Point, tet: (Point, Point, Point, Point)) -> bool {
    // determine whether p lies within tet's circumsphere
    let (a, b, c, d) = tet;

    // TODO replace with faster / more accurate 4x4 determinant
    let mat = ConstMatrix::<25>::from_flat_array(
        5,
        [
            a[0],
            a[1],
            a[2],
            a[0].powi(2) + a[1].powi(2) + a[1].powi(2),
            1.0,
            b[0],
            b[1],
            b[2],
            b[0].powi(2) + b[1].powi(2) + b[1].powi(2),
            1.0,
            c[0],
            c[1],
            c[2],
            c[0].powi(2) + c[1].powi(2) + c[1].powi(2),
            1.0,
            d[0],
            d[1],
            d[2],
            d[0].powi(2) + d[1].powi(2) + d[1].powi(2),
            1.0,
            p[0],
            p[1],
            p[2],
            p[0].powi(2) + p[1].powi(2) + p[1].powi(2),
            1.0,
        ],
    );

    if let Ok(val) = mat.determinant() {
        val > 0.0
    } else {
        false
    }
}

// a line, represented as ax + by = c
#[derive(Clone, Copy, Debug)]
struct Line {
    a: f64,
    b: f64,
    c: f64,
}

impl Line {
    fn from_segment(p: Point, q: Point) -> Self {
        let a = q[1] - p[1];
        let b = p[0] - q[0];
        let c = a * p[0] + b * p[1];
        Line { a, b, c }
    }

    fn perp_bisect(p: Point, q: Point) -> Self {
        // contruct a perpendicular bisector of pq
        let along = Self::from_segment(p, q);

        let mid = p.mid(q);

        let c = -1.0 * along.b * mid[0] + along.a * mid[1];
        let a = -1.0 * along.b;
        let b = along.a;
        Line { a, b, c }
    }

    fn intersect(&self, other: Line) -> Option<Point> {
        // find the intersection of two lines
        let det = self.a * other.b - other.a * self.b;
        if det == 0.0 {
            None
        } else {
            let x = (other.b * self.c - self.b * other.c) / det;
            let y = (self.a * other.c - other.a * self.c) / det;
            Some((x, y).into())
        }
    }
}

pub fn circumcenter(tri: (Point, Point, Point)) -> Option<Point> {
    // returns None if triangle is degenerate

    let (p, q, r) = tri;
    let leg_a = Line::perp_bisect(p, q);
    let leg_b = Line::perp_bisect(q, r);

    leg_a.intersect(leg_b)
}
