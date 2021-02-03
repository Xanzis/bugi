use std::convert::{From, TryFrom, TryInto};
use std::fmt;
use std::ops::{Index, IndexMut};

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
        SpatialError {msg: msg.to_string()}
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
        if n > 3 { panic!("dimension too high") }
        for i in 0..n {
            data[i] = vals[i];
        }
        Point {n, data}
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.dim() {
            1 => write!(f, "({:1.4})", self.data[0]),
            2 => write!(f, "({:1.4}, {:1.4})", self.data[0], self.data[1]),
            3 => write!(f, "({:1.4}, {:1.4}, {:1.4})", self.data[0], self.data[1], self.data[2]),
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
            _ => Err(SpatialError::new("bad dimensions in conversion"))
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
            _ => Err(SpatialError::new("bad dimensions in conversion"))
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
            _ => Err(SpatialError::new("bad dimensions in conversion"))
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
}
