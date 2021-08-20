use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;

use image::{Rgb, RgbImage};

mod bresenham;
pub mod color;
mod fill;

#[cfg(test)]
mod tests;

const DEFAULT_IMG_SIZE: u32 = 256;
const DOT_SIZE: u32 = 1;

pub struct VisOptions {
    color_map: Option<Box<dyn Fn(f64) -> Rgb<u8>>>,
    im_size: Option<u32>,
    show_mesh: bool,
}

impl VisOptions {
    pub fn new() -> Self {
        Self {
            color_map: None,
            im_size: None,
            show_mesh: true,
        }
    }

    pub fn color_map(mut self, map: Box<dyn Fn(f64) -> Rgb<u8>>) -> Self {
        self.color_map = Some(map);
        self
    }

    pub fn im_size(mut self, size: u32) -> Self {
        self.im_size = Some(size);
        self
    }

    pub fn show_mesh(mut self, x: bool) -> Self {
        self.show_mesh = x;
        self
    }
}

impl From<()> for VisOptions {
    fn from(_x: ()) -> Self {
        Self::new()
    }
}

// big ol' struct with all the stuff we're gonna need
pub struct Visualizer {
    dim: usize,
    im_size: u32,
    points: Vec<Point>,
    colors: Vec<usize>,
    edges: Option<Vec<(usize, usize)>>,
    triangles: Option<Vec<(usize, usize, usize)>>,
    node_vals: Option<Vec<f64>>,
    val_range: Option<(f64, f64)>,
}

impl Visualizer {
    pub fn add_points(&mut self, points: Vec<Point>, color: usize) {
        // TODO hacky hacky hacky rethink this stuff
        if self.node_vals.is_some() {
            panic!("can't add points to assemblage with defined node values")
        }

        self.colors.extend(vec![color; points.len()]);
        self.points.extend(points);
    }

    pub fn set_vals(&mut self, mut vals: Vec<f64>) {
        // normalize the values to [0, 1] and set the visualizer field
        // TODO el<->vis interfaces need to be improved, value interpolation + extra point overlay should be ok
        if vals.len() != self.points.len() {
            panic!("node values must have same count as nodes");
        }

        let min = *vals
            .iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        let max = *vals
            .iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();

        self.val_range = Some((min, max));

        let range = max - min;

        vals.iter_mut()
            .for_each(|x| *x = ((*x - min) / range).clamp(0.0, 1.0));
        self.node_vals = Some(vals);
    }

    pub fn set_edges(&mut self, edges: Vec<(usize, usize)>) {
        self.edges = if !edges.is_empty() { Some(edges) } else { None };
    }

    pub fn set_triangles(&mut self, tris: Vec<(usize, usize, usize)>) {
        self.triangles = if !tris.is_empty() { Some(tris) } else { None };
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
        let (xs, ys): (Vec<_>, Vec<_>) = coors.into_iter().unzip();

        let x_max = xs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let x_min = xs.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let y_max = ys.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let y_min = ys.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let max_range = if x_range > y_range { x_range } else { y_range };

        // want to bring the range down to 80% of the image width
        let target_range = 0.8 * (self.im_size as f64);
        let scaling = target_range / max_range;

        let middle_x = (x_max + x_min) / 2.0;
        let middle_y = (y_max + y_min) / 2.0;

        let mut pix_points = Vec::new();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let x_new = ((x - middle_x) * scaling + (self.im_size as f64) / 2.0).round();
            // flip y so the image appears in the familiar x/y orientation
            let y_new = (-1.0 * (y - middle_y) * scaling + (self.im_size as f64) / 2.0).round();
            pix_points.push((x_new as u32, y_new as u32));
        }

        pix_points
    }

    pub fn draw<T>(&mut self, fileloc: &str, options: T)
    where
        T: Into<VisOptions>,
    {
        let options = options.into();

        if let Some(x) = options.im_size {
            self.im_size = x;
        }

        let pix_points = self.enpixel();
        let mut img = RgbImage::from_pixel(self.im_size, self.im_size, color::BACK);

        // draw triangles if present
        if let (Some(triangles), Some(node_vals)) = (self.triangles.clone(), self.node_vals.clone())
        {
            for t in triangles.iter() {
                let (a, b, c) = *t;
                let tri = [pix_points[a], pix_points[b], pix_points[c]];
                let vals = [node_vals[a], node_vals[b], node_vals[c]];
                let to_draw = fill::triangle_interp(tri, vals);

                for (loc, v) in to_draw.into_iter() {
                    if let Some(cm) = options.color_map.as_ref() {
                        img.put_pixel(loc.0, loc.1, cm(v));
                    } else {
                        img.put_pixel(loc.0, loc.1, color::hot_map(v));
                    }
                }
            }
        }

        if options.show_mesh {
            // draw lines
            if let Some(edges) = &self.edges {
                for e in edges.iter() {
                    let orig = pix_points.get(e.0);
                    let end = pix_points.get(e.1);
                    if let (Some(o), Some(e)) = (orig, end) {
                        let to_draw = bresenham::line_unsigned(o, e);
                        for d in to_draw {
                            img.put_pixel(d.0, d.1, Rgb([0, 255, 0]))
                        }
                    }
                }
            }

            // draw points
            for ((x, y), c) in pix_points.into_iter().zip(self.colors.iter()) {
                let to_draw = fill::dot_points(x, y);
                for (i, j) in to_draw.into_iter() {
                    img.put_pixel(i, j, color::STDCOL[*c]);
                }
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
            im_size: DEFAULT_IMG_SIZE,
            points,
            colors: vec![0; n],
            edges: None,
            triangles: None,
            node_vals: None,
            val_range: None,
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
