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

impl TryInto<f64> for Point {
    type Error = SpatialError;
    fn try_into(self) -> Result<f64, SpatialError> {
        if let Point::One(x) = self {
            Ok(x)
        }
        else {
            Err(SpatialError::new("mismatched dimensions for conversion"))
        }
    }
}

impl From<(f64, f64)> for Point {
    fn from(x: (f64, f64)) -> Self {
        Point::Two(x.0, x.1)
    }
}

impl TryInto<(f64, f64)> for Point {
    type Error = SpatialError;
    fn try_into(self) -> Result<(f64, f64), SpatialError> {
        if let Point::Two(x, y) = self {
            Ok((x, y))
        }
        else {
            Err(SpatialError::new("mismatched dimensions for conversion"))
        }
    }
}

impl From<(f64, f64, f64)> for Point {
    fn from(x: (f64, f64, f64)) -> Self {
        Point::Thr(x.0, x.1, x.2)
    }
}

impl TryInto<(f64, f64, f64)> for Point {
    type Error = SpatialError;
    fn try_into(self) -> Result<(f64, f64, f64), SpatialError> {
        if let Point::Thr(x, y, z) = self {
            Ok((x, y, z))
        }
        else {
            Err(SpatialError::new("mismatched dimensions for conversion"))
        }
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

// TODO this shows Point would better be implemented as a usize, [f64; 3] pair
impl Index<usize> for Point {
    type Output = f64;
    fn index(&self, i: usize) -> Self::Output {
        match self.clone() {
            Point::One(x) => {
                if i >= 1 { panic!("index {} out of range (max 0)", i) }
                else { x }
            },
            Point::Two(x, y) => {
                if i >= 2 { panic!("index {} out of range (max 1)", i) }
                else if i == 0 { x }
                else { y }
            },
            Point::Three(x, y, z) => {
                if i >= 3 { panic!("index {} out of range (max 2)", i) }
                else if i == 0 { x }
                else if i == 1 { y }
                else { z }
            },
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
