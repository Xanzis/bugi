use super::Point;
use std::convert::TryInto;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Orient {
    Right,
    Left,
    Line,
}

pub fn triangle_dir(p: Point, q: Point, r: Point) -> Orient {
    // return the direction triangle (p, q, r) turns
    let val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);
    if val == 0.0 {
        Orient::Line
    } else if val > 0.0 {
        Orient::Right
    } else {
        Orient::Left
    }
}

pub fn jarvis_hull(points: &Vec<Point>) -> Vec<usize> {
    // find the indices of a counterclockwise hull around the supplied points
    let mut res = Vec::new();
    let n = points.len();
    if n < 3 {
        return res;
    }

    // find the leftmost point
    let l = (0..n)
        .min_by(|x, y| points[*x][0].partial_cmp(&points[*y][0]).unwrap())
        .unwrap();

    let mut p = l;

    loop {
        res.push(p);
        let mut q = (p + 1) % n;

        for i in 0..n {
            // find the rightmost point
            match triangle_dir(points[p], points[i], points[q]) {
                Orient::Left => q = i,
                Orient::Right => (),
                Orient::Line => (),
            };
        }

        p = q;
        if p == l {
            return res;
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
    0.0 <= a && 1.0 >= a && 0.0 <= b && 1.0 >= b && 0.0 <= c && 1.0 >= c
}
