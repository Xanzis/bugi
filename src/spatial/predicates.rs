use std::convert::TryInto;

use crate::matrix::{Inverse, LinearMatrix, MatrixLike};

use super::Point;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orient {
    Negative,
    Positive,
    Zero,
}

pub fn triangle_dir(tri: (Point, Point, Point)) -> Orient {
    // return whether the triangle (p, q, r) turns counterclockwise
    // postive is natural (ccw), negative is cw
    let (p, q, r) = tri;
    let val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);
    if val == 0.0 {
        Orient::Zero
    } else if val > 0.0 {
        Orient::Negative
    } else {
        Orient::Positive
    }
}

pub fn tetrahedron_dir(tet: (Point, Point, Point, Point)) -> Orient {
    // return the orientation of a tetrahedron
    let (a, b, c, d) = tet;

    let mat = LinearMatrix::from_flat(
        4,
        vec![
            a[0], a[1], a[2], 1.0, b[0], b[1], b[2], 1.0, c[0], c[1], c[2], 1.0, d[0], d[1], d[2],
            1.0,
        ],
    );

    let val = mat.determinant();
    if val == 0.0 {
        Orient::Zero
    } else if val > 0.0 {
        Orient::Positive
    } else {
        Orient::Negative
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
    0.0 <= a && 1.0 >= a && 0.0 <= b && 1.0 >= b && 0.0 <= c && 1.0 >= c
}

pub fn in_circle(p: Point, tri: (Point, Point, Point)) -> bool {
    // determine whether p lies within tri's circumcircle
    // tri's points should be in counterclockwise order
    let (a, b, c) = tri;

    // TODO replace with faster / more accurate 3x3 determinant
    let mat = LinearMatrix::from_flat(
        4,
        vec![
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

    // TODO make sure determinant returns 0.0 when mat is singular
    mat.determinant() > 0.0
}

pub fn in_sphere(p: Point, tet: (Point, Point, Point, Point)) -> bool {
    // determine whether p lies within tet's circumsphere
    let (a, b, c, d) = tet;

    // TODO replace with faster / more accurate 4x4 determinant
    let mat = LinearMatrix::from_flat(
        5,
        vec![
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

    // TODO same check as above
    mat.determinant() > 0.0
}

fn on_segment(seg: (Point, Point), q: Point) -> bool {
    // given colinear, 2D p q and r, find if q lies on pr (seg)
    let (p, r) = seg;
    q[0] <= p[0].max(r[0])
        && q[0] >= p[0].min(r[0])
        && q[1] <= p[1].max(r[1])
        && q[1] >= p[1].min(r[1])
}

pub fn segments_intersect(a: (Point, Point), b: (Point, Point)) -> bool {
    // test whether line segments a and b intersect
    let l = triangle_dir((a.0, a.1, b.0));
    let m = triangle_dir((a.0, a.1, b.1));
    let n = triangle_dir((b.0, b.1, a.0));
    let o = triangle_dir((b.0, b.1, a.1));

    if (l != m) && (n != o) {
        true
    } else {
        if l == Orient::Zero && on_segment(a, b.0) {
            true
        } else if m == Orient::Zero && on_segment(a, b.1) {
            true
        } else if n == Orient::Zero && on_segment(b, a.0) {
            true
        } else if o == Orient::Zero && on_segment(b, a.1) {
            true
        } else {
            false
        }
    }
}
