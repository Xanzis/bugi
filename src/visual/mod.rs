use crate::spatial::Point;
use crate::matrix::Matrix;

use image::{RgbImage, Rgb};

const IMG_SIZE: u32 = 64;

// big ol' struct with all the stuff we're gonna need
struct Assemblage {
	dim: usize,
	points: Vec<Point>,
	proj_points: Option<Vec<(f64, f64)>>,
	pix_points: Option<Vec<(u32, u32)>>,
	edges: Option<Vec<usize>>,
}

impl Assemblage {

	fn project(&mut self) {
		// fill out projections of the points onto the 2d plane
		let mut projs: Vec<(f64, f64)> = Vec::new();
		match self.dim {
			1 => {
				for p in self.points.iter() {
					if let Point::One(x) = p {
						projs.push((*x, 0.0));
					}
					else {
						panic!("bad point variant");
					}
				}
			},
			2 => {
				for p in self.points.iter() {
					if let Point::Two(x, y) = p {
						projs.push((*x, *y));
					}
					else {
						panic!("bad point variant");
					}
				}
			},
			3 => {
				projs.extend(project(self.points.clone()));
			},
			_ => panic!("bad dim"),
		}

		self.proj_points = Some(projs);
	}

	fn enpixel(&mut self) {
		// calculate the pixel positions of the assemblage points
		if let Some(coors) = &self.proj_points {
			let n_points = coors.len() as f64;

			let mut x_sum = 0.0;
			let mut x_max = f64::NEG_INFINITY;
			let mut x_min = f64::INFINITY;

			let mut y_sum = 0.0;
			let mut y_max = f64::NEG_INFINITY;
			let mut y_min = f64::INFINITY;

			for (x, y) in coors.iter().cloned() {
				x_sum += x;
				if x > x_max { x_max = x; }
				if x < x_min { x_min = x; }

				y_sum += y;
				if y > y_max { y_max = y; }
				if y < y_min { y_min = y; }
			}

			let x_mean = x_sum / n_points;
			let y_mean = y_sum / n_points;

			let x_range = x_max - x_min;
			let y_range = y_max - y_min;

			let max_range = if x_range > y_range { x_range } else { y_range };

			// want to bring the range down to 80% of the image width
			let target_range = 0.8 * (IMG_SIZE as f64);
			let scaling = target_range / max_range;

			let mut pix_points: Vec<(u32, u32)> = Vec::new();
			for (x, y) in coors.iter().cloned() {
				let x_new = ((x - x_mean) * scaling + (IMG_SIZE as f64) / 2.0).round();
				let y_new = ((y - y_mean) * scaling + (IMG_SIZE as f64) / 2.0).round();
				pix_points.push((x_new as u32, y_new as u32));
			}
			self.pix_points = Some(pix_points);
		}
	}

	fn draw(&mut self, fileloc: &str) {
		if self.pix_points == None {
			if self.proj_points == None {
				self.project();
			}
			self.enpixel();
		}
		let mut img = RgbImage::new(IMG_SIZE, IMG_SIZE);

		// draw points -- TODO draw thicker points
		for (x, y) in self.pix_points.as_ref().unwrap().iter().cloned() {
			img.put_pixel(x, y, Rgb([255, 0, 0]));
		}

		img.save(fileloc).unwrap();
	}
}

impl From<Vec<Point>> for Assemblage {
	fn from(points: Vec<Point>) -> Self {
		let dim = points[0].dim();
		if points.iter().any(|x| x.dim() != dim) {
			panic!("inconsistent dimensionalities in Assemblage conversion");
		}
		Assemblage {dim, points, proj_points: None, pix_points: None, edges: None}
	}
}

fn rotation_matrix(x_ang: f64, y_ang: f64) -> Matrix {
	let sin_xa = x_ang.sin();
	let cos_xa = x_ang.cos();

	let sin_ya = y_ang.sin();
	let cos_ya = y_ang.cos();

	let a = Matrix::from_rows( vec![vec![1.0, 0.0, 0.0], 
									vec![0.0, cos_xa, -sin_xa], 
									vec![0.0, sin_xa, cos_xa]]);

	let b = Matrix::from_rows( vec![vec![cos_ya, 0.0, sin_ya], 
									vec![0.0, 1.0, 0.0], 
									vec![-sin_ya, 0.0, cos_ya]]);

	&a * &b	
}

fn project(points: Vec<Point>) -> Vec<(f64, f64)> {
	// compute a parallel projection at a slightly interesting angle
	let r = rotation_matrix(0.79, 0.5);

	let points = Matrix::from_points_row(points);
	let points = points.transpose();
	if points.shape().0 != 3 { panic!("bad dimension count in visual::project"); }

	let new_points = &r * &points;
	// return the x and y coordinates of the new points by zipping the first two rows
	new_points.row(0)
		.cloned()
		.zip(new_points.row(1).cloned())
		.collect()
}

#[cfg(test)]
mod tests {
	use crate::spatial::Point;
	use super::Assemblage;

    #[test]
    fn one_d() {
        let mut asm: Assemblage = vec![Point::One(1.0), Point::One(3.5)].into();
        asm.draw("test_generated/one.png");
    }
    #[test]
    fn two_d() {
    	let mut asm: Assemblage = vec![Point::Two(1.0, 2.0), Point::Two(-25.0, 37.0), Point::Two(12.0, -5.0)].into();
    	asm.draw("test_generated/two.png");
    }
    #[test]
    fn three_d() {
    	let mut asm: Assemblage = vec![
    		Point::Thr(-1.0, 1.0, 1.0), Point::Thr(1.0, 1.0, 1.0), Point::Thr(1.0, -1.0, 1.0), Point::Thr(-1.0, -1.0, 1.0),
    		Point::Thr(-1.0, 1.0, -1.0), Point::Thr(1.0, 1.0, -1.0), Point::Thr(1.0, -1.0, -1.0), Point::Thr(-1.0, -1.0, -1.0)].into();
    	asm.draw("test_generated/three.png");
    }
}