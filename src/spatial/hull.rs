use super::Point;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Orient {
    Right,
    Left,
    Line,
}

pub fn triangle_dir(p: Point, q: Point, r: Point) -> Orient {
    // return the direction triangle (p, q, r) turns
    let val = (q[1] - p[1]) * (r[0] - q[0]) -
              (q[0] - p[0]) * (r[1] - q[1]);
    if val == 0.0 { Orient::Line }
    else if val > 0.0 { Orient::Right }
    else { Orient::Left }
}

pub fn jarvis_hull(points: &Vec<Point>) -> Vec<usize> {
    // find the indices of a counterclockwise hull around the supplied points
    let mut res = Vec::new();
    let n = points.len();
    if n < 3 { return res }

    // find the leftmost point
    let l = (0..n)
        .min_by(
            |x, y| points[*x][0].partial_cmp(&points[*y][0]).unwrap()
            )
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
        if p == l { return res }
    }
}