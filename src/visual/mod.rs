use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;

use image::{Rgb, RgbImage};

mod bresenham;

#[cfg(test)]
mod tests;

const IMG_SIZE: u32 = 128;
const DOT_SIZE: u32 = 1;
const COLORS: [Rgb<u8>; 4] = [Rgb([255, 0, 0]), Rgb([255, 255, 0]), Rgb([0, 0, 255]), Rgb([255, 0, 255])];

// big ol' struct with all the stuff we're gonna need
pub struct Visualizer {
    dim: usize,
    points: Vec<Point>,
    colors: Vec<usize>,
    edges: Option<Vec<(usize, usize)>>,
}

impl Visualizer {
    pub fn add_points(&mut self, points: Vec<Point>, color: usize) {
        self.colors.extend(vec![color; points.len()]);
        self.points.extend(points);
    }

    fn project(&self) -> Vec<(f64, f64)> {
        // fill out projections of the points onto the 2d plane
        let mut projs = Vec::new();
        match self.dim {
            1 => {
                for p in self.points.iter() {
                    projs.push((p[0], 0.0));
                }
            }
            2 => {
                for p in self.points.iter() {
                    projs.push((p[0], p[1]));
                }
            }
            3 => {
                projs.extend(project(self.points.clone()));
            }
            _ => panic!("bad dim"),
        }

        projs
    }

    fn enpixel(&self) -> Vec<(u32, u32)> {
        // calculate the pixel positions of the Visualizer points
        let coors = self.project();
        let n_points = coors.len() as f64;
        let (xs, ys): (Vec<_>, Vec<_>) = coors.into_iter().unzip();

        let x_max = xs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let x_min = xs.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let y_max = ys.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let y_min = ys.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        let x_sum: f64 = xs.iter().sum();
        let y_sum: f64 = ys.iter().sum();

        let x_mean = x_sum / n_points;
        let y_mean = y_sum / n_points;

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let max_range = if x_range > y_range { x_range } else { y_range };

        // want to bring the range down to 80% of the image width
        let target_range = 0.8 * (IMG_SIZE as f64);
        let scaling = target_range / max_range;

        let middle_x = (x_max + x_min) / 2.0;
        let middle_y = (y_max + y_min) / 2.0;

        let mut pix_points = Vec::new();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let x_new = ((x - middle_x) * scaling + (IMG_SIZE as f64) / 2.0).round();
            // flip y so the image appears in the familiar x/y orientation
            let y_new = (-1.0 * (y - middle_y) * scaling + (IMG_SIZE as f64) / 2.0).round();
            pix_points.push((x_new as u32, y_new as u32));
        }

        pix_points
    }

    pub fn draw(&mut self, fileloc: &str) {
        let pix_points = self.enpixel();
        let mut img = RgbImage::new(IMG_SIZE, IMG_SIZE);

        // draw lines
        if let Some(edges) = &self.edges {
            for e in edges.iter() {
                let orig = pix_points.get(e.0);
                let end = pix_points.get(e.1);
                if let (Some(o), Some(e)) = (orig, end) {
                    let to_draw = bresenham::line_unsigned(orig.unwrap(), end.unwrap());
                    for d in to_draw {
                        img.put_pixel(d.0, d.1, Rgb([0, 255, 0]))
                    }
                }
            }
        }

        // draw points -- TODO draw thicker points
        for ((x, y), c) in pix_points.into_iter().zip(self.colors.iter()) {
            let to_draw = dot_points(x, y);
            for (i, j) in to_draw.into_iter() {
                img.put_pixel(i, j, COLORS[*c]);   
            }
        }

        img.save(fileloc).unwrap();
    }
}

impl From<Vec<Point>> for Visualizer {
    fn from(points: Vec<Point>) -> Self {
        let dim = points[0].dim();
        let n = points.len();
        if points.iter().any(|x| x.dim() != dim) {
            panic!("inconsistent dimensionalities in Visualizer conversion");
        }
        Visualizer {
            dim,
            points,
            colors: vec![0; n],
            edges: None,
        }
    }
}

fn rotation_matrix(x_ang: f64, y_ang: f64) -> LinearMatrix {
    let sin_xa = x_ang.sin();
    let cos_xa = x_ang.cos();

    let sin_ya = y_ang.sin();
    let cos_ya = y_ang.cos();

    let a = LinearMatrix::from_rows(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, cos_xa, -sin_xa],
        vec![0.0, sin_xa, cos_xa],
    ]);

    let b = LinearMatrix::from_rows(vec![
        vec![cos_ya, 0.0, sin_ya],
        vec![0.0, 1.0, 0.0],
        vec![-sin_ya, 0.0, cos_ya],
    ]);

    a.mul(&b)
}

fn project(points: Vec<Point>) -> Vec<(f64, f64)> {
    // compute a parallel projection at a slightly interesting angle
    let r = rotation_matrix(0.79, 0.5);

    let mut points = LinearMatrix::from_points_row(points);
    points.transpose();
    if points.shape().0 != 3 {
        panic!("bad dimension count in visual::project");
    }

    let new_points: LinearMatrix = r.mul(&points);
    // return the x and y coordinates of the new points by zipping the first two rows
    new_points
        .row(0)
        .cloned()
        .zip(new_points.row(1).cloned())
        .collect()
}

fn dot_points(center_x: u32, center_y: u32) -> Vec<(u32, u32)> {
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