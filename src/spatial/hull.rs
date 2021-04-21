use super::predicates::{triangle_dir, Orient};
use super::Point;

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
            match triangle_dir((points[p], points[i], points[q])) {
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
