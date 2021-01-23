use std::fmt;
use std::convert::{From, TryFrom};

#[derive(Debug, PartialEq, Clone)]
pub enum Point {
	One(f64),
	Two(f64, f64),
	Thr(f64, f64, f64),
}

impl Point {
	pub fn dim(&self) -> usize {
		match self {
			Point::One(_) => 1,
			Point::Two(_, _) => 2,
			Point::Thr(_, _, _) => 3,
		}
	}
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    	match self {
    		Point::One(x) => write!(f, "({:1.4})", x),
    		Point::Two(x, y) => write!(f, "({:1.4}, {:1.4})", x, y),
    		Point::Thr(x, y, z) => write!(f, "({:1.4}, {:1.4}, {:1.4})", x, y, z),
    	}
    }
}

impl From<f64> for Point {
	fn from(x: f64) -> Self {
		Point::One(x)
	}
}

impl From<(f64, f64)> for Point {
	fn from(x: (f64, f64)) -> Self {
		Point::Two(x.0, x.1)
	}
}

impl From<(f64, f64, f64)> for Point {
	fn from(x: (f64, f64, f64)) -> Self {
		Point::Thr(x.0, x.1, x.2)
	}
}

impl TryFrom<Vec<f64>> for Point {
	type Error = &'static str;

	fn try_from(x: Vec<f64>) -> Result<Self, Self::Error> {
		match x.len() {
			1 => Ok(Point::One(x[0])),
			2 => Ok(Point::Two(x[0], x[1])),
			3 => Ok(Point::Thr(x[0], x[1], x[2])),
			_ => Err("unsupported dimensionality for Point"),
		}
	}
}

impl From<Point> for Vec<f64> {
	fn from(x: Point) -> Self {
		match x {
			Point::One(x) => vec![x],
			Point::Two(x, y) => vec![x, y],
			Point::Thr(x, y, z) => vec![x, y, z],
		}
	}
}

#[cfg(test)]
mod tests {
	use super::Point;
	use std::convert::TryFrom;

	#[test]
	fn make_one() {
		assert!(Point::One(1.2) == Point::from(1.2))
	}
	#[test]
	fn make_two() {
		assert!(Point::Two(1.2, 2.3) == Point::from((1.2, 2.3)))
	}
	#[test]
	fn make_three() {
		assert!(Point::Thr(1.2, 2.3, 3.4) == Point::from((1.2, 2.3, 3.4)))
	}
	#[test]
	fn point_disp() {
		let goal = "(0.0000)";
		assert!(format!("{}", Point::One(0.0)) == goal)
	}
	#[test]
	fn good_try() {
		let test = Point::try_from(vec![1.0, 2.0, 3.0]);
		assert!(test == Ok(Point::Thr(1.0, 2.0, 3.0)))
	}
	#[test]
	fn bad_try() {
		let test = Point::try_from(vec![1.0, 2.0, 3.0, 4.0]);
		assert!(test.is_err())
	}
}