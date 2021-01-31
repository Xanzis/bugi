use super::{MatrixShape, MatrixLike};
use std::cmp::PartialEq;
use std::ops::{Add, Sub, AddAssign, MulAssign};
use std::fmt;

#[derive(Clone, Debug)]
pub struct LinearMatrix {
    dims: (usize, usize),
    row_maj: bool,
    data: Vec<f64>,
}

impl LinearMatrix {
    fn pos(&self, loc: (usize, usize)) -> Option<usize> {
        // find the location of the desired element in data
        // read differently for row/column major orders of buffer
        if (loc.1 >= self.dims.1) || (loc.0 >= self.dims.0) { return None }
        if self.row_maj {
            Some((self.dims.1 * loc.0) + loc.1)
        }
        else {
            Some((self.dims.0 * loc.1) + loc.0)
        }
    }
}

impl MatrixLike for LinearMatrix {
    fn shape(&self) -> (usize, usize) {
        self.dims
    }

    fn get(&self, loc: (usize, usize)) -> Option<&f64> {
        if let Some(i) = self.pos(loc) { self.data.get(i) } else { None }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        if let Some(i) = self.pos(loc) { self.data.get_mut(i) } else { None }
    }

    fn transpose(&mut self) {
        self.row_maj = !self.row_maj;
        self.dims = (self.dims.1, self.dims.0);
    }

    fn zeros<T: Into<MatrixShape>>(shape: T) -> Self {
        let shape: MatrixShape = shape.into();
        let dims = (shape.nrow, shape.ncol);
        let data = vec![0.0; dims.0 * dims.1];
        Self { dims, row_maj: true, data }
    }
    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(shape: T, data: U) -> Self {
        // turn a row-major vector of values into a matrix
        let shape: MatrixShape = shape.into();
        let dims = (shape.nrow, shape.ncol);
        let data: Vec<f64> = data.into_iter().collect();
        if (dims.0 * dims.1) != data.len() {
            panic!("bad shape {:?} for data length {}", dims, data.len())
        }
        Self { dims, row_maj: true, data }
    }
}

impl PartialEq for LinearMatrix {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() { return false }
        if self.row_maj == other.row_maj {
            // data is arranged the same
            return self.data == other.data
        }
        // if the data orders don't align, check the slow way
        let (nrow, ncol) = self.shape();
        for i in 0..nrow {
            for j in 0..ncol {
                if self.get((i, j)) != other.get((i, j)) { return false }
            }
        }
        true
    }
}

impl fmt::Display for LinearMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (nrow, ncol) = self.shape();
        write!(f, "rows: {} cols: {}\n", nrow, ncol)?;
        for i in 0..nrow {
            for j in 0..ncol {
                write!(f, "{:1.5} ", self.get((i, j)).unwrap())?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl<'a> Add<&'a LinearMatrix> for &'a LinearMatrix {
    type Output = LinearMatrix;
    fn add(self, rhs: Self) -> Self::Output {
        if self.dims != rhs.dims { panic!("incompatible shapes") }
        if self.row_maj == rhs.row_maj {
            let new_data = self.data.iter()
                                    .zip(rhs.data.iter())
                                    .map(|(x, y)| x + y)
                                    .collect();
            LinearMatrix {dims: self.dims, row_maj: self.row_maj, data: new_data}
        }
        else {
            let mut res = LinearMatrix {dims: self.dims, row_maj: true, data: Vec::new()};
            for i in 0..self.dims.0 {
                res.data.extend(self.row(i)
                                    .zip(rhs.row(i))
                                    .map(|(x, y)| x + y));
            }
            res
        }
    }
}

impl<'a> Sub<&'a LinearMatrix> for &'a LinearMatrix {
    type Output = LinearMatrix;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.dims != rhs.dims { panic!("incompatible shapes") }
        if self.row_maj == rhs.row_maj {
            let new_data = self.data.iter()
                                    .zip(rhs.data.iter())
                                    .map(|(x, y)| x - y)
                                    .collect();
            LinearMatrix {dims: self.dims, row_maj: self.row_maj, data: new_data}
        }
        else {
            let mut res = LinearMatrix {dims: self.dims, row_maj: true, data: Vec::new()};
            for i in 0..self.dims.0 {
                res.data.extend(self.row(i)
                                    .zip(rhs.row(i))
                                    .map(|(x, y)| x - y));
            }
            res
        }
    }
}

impl AddAssign<f64> for LinearMatrix {
    fn add_assign(&mut self, other: f64) {
        self.data.iter_mut().for_each(|x| *x += other);
    }
}

impl MulAssign<f64> for LinearMatrix {
    fn mul_assign(&mut self, other: f64) {
        self.data.iter_mut().for_each(|x| *x *= other);
    }
}