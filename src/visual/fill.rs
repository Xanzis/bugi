use super::DOT_SIZE;
use crate::spatial::{hull, Point};

pub fn dot_points(center_x: u32, center_y: u32) -> Vec<(u32, u32)> {
    // returns the points to draw for a dot
    // just a boring ol' square, might change to a more circular thing
    // TODO maybe handle *edge* cases better haha get it
    let mut res = Vec::new();
    for x in (center_x - DOT_SIZE)..=(center_x + DOT_SIZE) {
        for y in (center_y - DOT_SIZE)..=(center_y + DOT_SIZE) {
            res.push((x, y));
        }
    }
    res
}

pub fn triangle_interp(points: [(u32, u32); 3], vals: [f64; 3]) -> Vec<((u32, u32), f64)> {
    // generate a list of pixels within the given triangle, with interpolated node values
    let x_max = points.iter().map(|(x, _)| x).max().unwrap().clone();
    let x_min = points.iter().map(|(x, _)| x).min().unwrap().clone();
    let y_max = points.iter().map(|(_, y)| y).max().unwrap().clone();
    let y_min = points.iter().map(|(_, y)| y).min().unwrap().clone();

    let tri: (Point, Point, Point) = (points[0].into(), points[1].into(), points[2].into());
    let mut res = Vec::new();

    for x in x_min..=x_max {
        for y in y_min..=y_max {
            let p: Point = (x, y).into();
            if hull::in_triangle(p, tri) {
                let weights = hull::bary_coor(p, tri);
                let weighted_val = weights.0 * vals[0] + weights.1 * vals[1] + weights.2 * vals[2];
                res.push(((x, y), weighted_val));
            }
        }
    }
    res
}
