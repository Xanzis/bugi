use super::{MatrixData, MatrixShape};
use std::cmp::PartialEq;
use std::ops::{Add, Mul, Sub};

pub enum Buffer {
    Linear(LinearBuffer),
}

pub struct LinearBuffer {
    dims: (usize, usize),
    row_maj: bool,
    data: Vec<f64>,
}

impl LinearBuffer {
    fn pos(&self, loc: (usize, usize)) -> Option<usize> {
        // find the location of the desired element in data
        // read differently for row/column major orders of buffer
        if (loc.1 >= self.dims.1) || (loc.0 >= self.dims.0) { return None }
        if self.row_maj {
            Some((self.dims.0 * loc.0) + loc.1)
        }
        else {
            Some((self.dims.1 * loc.1) + loc.0)
        }
    }
}

impl MatrixData for LinearBuffer {
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
    }

    fn zeros(shape: MatrixShape) -> LinearBuffer {
        let dims = (shape.nrow, shape.ncol);
        let data = vec![0.0; dims.0 * dims.1];
        LinearBuffer { dims, row_maj: true, data }
    }
    fn from_flat(shape: MatrixShape, data: Vec<f64>) -> LinearBuffer {
        // turn a row-major vector of values into a matrix
        let dims = (shape.nrow, shape.ncol);
        if (dims.0 * dims.1) != data.len() {
            panic!("bad shape {} for data length {}", dims, data.len())
        }
        LinearBuffer{ dims, row_maj: true, data }
    }
}

impl Mul<&LinearBuffer> for &LinearBuffer{
    type Output = LinearBuffer;

    fn mul(self, rhs: Self) -> Self::Output {
        let a_shape = self.shape();
        let b_shape = rhs.shape();

        if a_shape.1 != b_shape.0 {
            panic!(
                "improper shapes for matrix multiplication: {:?} and {:?}",
                a_shape, b_shape
            )
        }

        let res_shape = (a_shape.0, b_shape.1);
        let mut res_vals = Vec::reserve(res_shape.0 * res_shape.1);

        for r in 0..res_shape.0 {
            for c in 0..res_shape.1 {
                let dot = self.row(r)
                    .zip(rhs.col(c))
                    .map(|x| x.0 * x.1).sum();
                res_vals.push(dot);
            }
        }
        Self::Output::from_flat(res_shape.into(), res_vals);
    }
}

impl Add<&LinearBuffer> for &LinearBuffer{
    type Output = LinearBuffer;

    fn add(self, rhs: Self) -> Self::Output {
        if self.shape() != rhs.shape() {
            panic!(
                "improper shapes for matrix addition: {:?} and {:?}",
                self.shape(),
                rhs.shape()
            )
        }

        // TODO this is a little generic and can be sped up for the buffer
        let res_shape = self.shape();
        let mut res_vals = Vec::reserved(res_shape.0 * res_shape.1);
        for r in 0..res_shape.0 {
            for c in 0..res_shape.1 {
                res_vals.push(self.get(r, c).unwrap() + rhs.get(r, c).unwrap());
            }
        }
        Self::Output::from_flat(res_shape.into(), res_vals);
    }
}

impl Sub<&LinearBuffer> for &LinearBuffer{
    type Output = LinearBuffer;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.shape() != rhs.shape() {
            panic!(
                "improper shapes for matrix subtraction: {:?} and {:?}",
                self.shape(),
                rhs.shape()
            )
        }

        let res_shape = self.shape();
        let mut res_vals = Vec::reserve(res_shape.0 * res_shape.1);
        for r in 0..res_shape.0 {
            for c in 0..res_shape.1 {
                res_vals.push(self.get(r, c).unwrap() - rhs.get(r, c).unwrap());
            }
        }
        Self::Output::from_flat(res_shape.into(), res_vals);
    }
}

impl PartialEq for LinearBuffer {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() { return false }
        if self.row_maj == other.row_maj {
            // data is arranged the same
            return self.data == other.data
        }
        // if the data orders don't align, check the slow way
        let (nrow, ncol) = self.shape();
        for i in 0..nrow {
            for j in 0..nrow {
                if self.get((i, j)) != other.get((i, j)) { return false }
            }
        }
        true
    }
}