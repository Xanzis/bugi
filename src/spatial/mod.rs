use std::cmp::min;
use std::convert::{From, TryFrom, TryInto};
use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

pub mod hull;

#[derive(Debug)]
pub struct SpatialError {
    msg: String,
}

impl fmt::Display for SpatialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpatialError: {}", self.msg)
    }
}

impl SpatialError {
    fn new(msg: &str) -> SpatialError {
        SpatialError {
            msg: msg.to_string(),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Point {
    n: usize,
    data: [f64; 3],
}

impl Point {
    pub fn dim(&self) -> usize {
        self.n
    }

    pub fn new(vals: &[f64]) -> Self {
        let mut data = [0.0; 3];
        let n = vals.len();
        if n > 3 {
            panic!("dimension too high")
        }
        for i in 0..n {
            data[i] = vals[i];
        }
        Point { n, data }
    }

    pub fn zero(n: usize) -> Self {
        Point { n, data: [0.0; 3] }
    }

    pub fn mid(&self, other: Point) -> Point {
        // find the midpoint between two points
        // TODO make sure nothing will set a nonzero value for an invalid dimension
        let data = [
            (self.data[0] + other.data[0]) / 2.0,
            (self.data[1] + other.data[1]) / 2.0,
            (self.data[2] + other.data[2]) / 2.0,
        ];
        let n = min(self.n, other.n);
        Point { n, data }
    }

    pub fn dist(&self, other: Point) -> f64 {
        // TODO make sure nothing will set a nonzero value for an invalid dimension
        (self.data[0].powi(2) + self.data[1].powi(2) + self.data[2].powi(2)).sqrt()
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.dim() {
            1 => write!(f, "({:1.4})", self.data[0]),
            2 => write!(f, "({:1.4}, {:1.4})", self.data[0], self.data[1]),
            3 => write!(
                f,
                "({:1.4}, {:1.4}, {:1.4})",
                self.data[0], self.data[1], self.data[2]
            ),
            _ => panic!("illegal point"),
        }
    }
}

impl From<f64> for Point {
    fn from(x: f64) -> Self {
        Point::new(&[x])
    }
}

impl TryInto<f64> for Point {
    type Error = SpatialError;
    fn try_into(self) -> Result<f64, SpatialError> {
        match self.dim() {
            1 => Ok(self[0]),
            _ => Err(SpatialError::new("bad dimensions in conversion")),
        }
    }
}

impl From<(f64, f64)> for Point {
    fn from(x: (f64, f64)) -> Self {
        Point::new(&[x.0, x.1])
    }
}

impl TryInto<(f64, f64)> for Point {
    type Error = SpatialError;
    fn try_into(self) -> Result<(f64, f64), SpatialError> {
        match self.dim() {
            2 => Ok((self[0], self[1])),
            _ => Err(SpatialError::new("bad dimensions in conversion")),
        }
    }
}

impl From<(f64, f64, f64)> for Point {
    fn from(x: (f64, f64, f64)) -> Self {
        Point::new(&[x.0, x.1, x.2])
    }
}

impl TryInto<(f64, f64, f64)> for Point {
    type Error = SpatialError;
    fn try_into(self) -> Result<(f64, f64, f64), SpatialError> {
        match self.dim() {
            3 => Ok((self[0], self[1], self[2])),
            _ => Err(SpatialError::new("bad dimensions in conversion")),
        }
    }
}

impl TryFrom<Vec<f64>> for Point {
    type Error = &'static str;

    fn try_from(x: Vec<f64>) -> Result<Self, Self::Error> {
        match x.len() {
            1 => Ok(Point::new(&[x[0]])),
            2 => Ok(Point::new(&[x[0], x[1]])),
            3 => Ok(Point::new(&[x[0], x[1], x[2]])),
            _ => Err("unsupported dimensionality for Point"),
        }
    }
}

impl From<Point> for Vec<f64> {
    fn from(x: Point) -> Self {
        match x.dim() {
            1 => vec![x[0]],
            2 => vec![x[0], x[1]],
            3 => vec![x[0], x[1], x[2]],
            _ => panic!("illegal point"),
        }
    }
}

impl Index<usize> for Point {
    type Output = f64;
    fn index(&self, i: usize) -> &Self::Output {
        if i >= self.dim() {
            panic!("index {} out of range for Point of dim {}", i, self.dim())
        }

        &self.data[i]
    }
}

impl IndexMut<usize> for Point {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        if i >= self.dim() {
            panic!("index {} out of range for Point of dim {}", i, self.dim())
        }

        &mut self.data[i]
    }
}

impl Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.dim() != other.dim() {
            panic!("point dimension mismatch")
        }
        let data = [
            self.data[0] + other.data[0],
            self.data[1] + other.data[1],
            self.data[2] + other.data[2],
        ];
        Point {
            n: self.dim(),
            data,
        }
    }
}

impl Sub for Point {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if self.dim() != other.dim() {
            panic!("point dimension mismatch")
        }
        let data = [
            self.data[0] - other.data[0],
            self.data[1] - other.data[1],
            self.data[2] - other.data[2],
        ];
        Point {
            n: self.dim(),
            data,
        }
    }
}

impl Mul<f64> for Point {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        let data = [
            self.data[0] * other,
            self.data[1] * other,
            self.data[2] * other,
        ];
        Point {
            n: self.dim(),
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Point;
    use std::convert::TryFrom;

    #[test]
    fn constructors() {
        assert!(Point::new(&[1.2]) == Point::from(1.2));
        assert!(Point::new(&[1.2, 2.3]) == Point::from((1.2, 2.3)));
        assert!(Point::new(&[1.2, 2.3, 3.4]) == Point::from((1.2, 2.3, 3.4)));
    }
    #[test]
    fn point_disp() {
        let goal = "(0.0000)";
        assert!(format!("{}", Point::new(&[0.0])) == goal)
    }
    #[test]
    fn good_try() {
        let test = Point::try_from(vec![1.0, 2.0, 3.0]);
        assert!(test == Ok(Point::new(&[1.0, 2.0, 3.0])))
    }
    #[test]
    fn bad_try() {
        let test = Point::try_from(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(test.is_err())
    }
    #[test]
    fn orientation() {
        use super::hull;
        use super::hull::Orient;
        let a = Point::new(&[-1.0, -0.3]);
        let b = Point::new(&[-0.5, 3.0]);
        let c = Point::new(&[10.0, -0.2]);
        assert_eq!(hull::triangle_dir(a, b, c), Orient::Right);
        assert_eq!(hull::triangle_dir(b, a, c), Orient::Left);
        assert_eq!(hull::triangle_dir(c, b, a), Orient::Left);
    }
    #[test]
    fn simple_hull() {
        use super::hull;
        let points = vec![
            Point::new(&[0.0, 2.0]),
            Point::new(&[3.0, 3.0]),
            Point::new(&[2.0, 1.0]),
            Point::new(&[1.0, 0.0]),
            Point::new(&[5.0, 0.0]),
        ];
        assert_eq!(hull::jarvis_hull(&points), vec![0, 3, 4, 1]);
    }
    #[test]
    fn point_arithmetic() {
        let p = Point::new(&[-1.0, 5.0]);
        assert_eq!(p - Point::new(&[1.0, 2.0]), Point::new(&[-2.0, 3.0]));
        assert_eq!(p + Point::new(&[1.0, 2.0]), Point::new(&[0.0, 7.0]));
        assert_eq!(p * 2.0, Point::new(&[-2.0, 10.0]));
    }
    #[test]
    fn point_in_triangle() {
        use super::hull;

        let p = Point::new(&[1.0, 1.0]);
        let q = Point::new(&[3.0, 1.0]);
        let a = Point::new(&[0.0, 1.0]);
        let b = Point::new(&[1.0, 2.0]);
        let c = Point::new(&[2.0, -1.0]);

        assert!(hull::in_triangle(p, (a, b, c)));
        assert!(hull::in_triangle(p, (c, b, a)));
        assert!(!hull::in_triangle(q, (a, b, c)));
        assert!(!hull::in_triangle(q, (c, b, a)));
    }
}
