use crate::spatial::Point;
use std::fmt;
use std::cmp::PartialEq;
use std::ops::{Add, Mul, Sub};

pub mod inverse;
pub mod norm;
pub mod buffer;

use inverse::Inverse;
use norm::Norm;

#[cfg(test)]
mod tests;

use buffer::{MatrixData, MatrixRow, MatrixCol};

// rolling my own (pretty limited) matrix math
// standard order for shape is (row, col)
// m[r][c] is stored at data[r*ncol + c]
#[derive(Clone, Debug)]
pub struct Matrix<T: MatrixData>(T);
pub type LinearMatrix = Matrix<buffer::LinearBuffer>;

// data type for matrix sizes
// will eventually add fields to help initialize sparse matrices
pub struct MatrixShape {
    ncol: usize,
    nrow: usize,
}

impl<T> fmt::Display for Matrix<T>
    where T: MatrixData
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mat_data = self.0.clone();
        let (nrow, ncol) = mat_data.shape();
        write!(f, "rows: {} cols: {}\n", nrow, ncol)?;
        for i in 0..nrow {
            for j in 0..ncol {
                write!(f, "{:1.5} ", mat_data.get((i, j)).unwrap())?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl From<(usize, usize)> for MatrixShape {
    fn from(dims: (usize, usize)) -> Self {
        MatrixShape{ nrow: dims.0, ncol: dims.1 }
    }
}

impl<T> Matrix<T>
    where T: MatrixData
{
    pub fn from_rows(data: Vec<Vec<f64>>) -> Self {
        let mut total = Vec::new();
        let nrow = data.len();
        if nrow == 0 { panic!("empty data") }
        let ncol = data.get(0).unwrap().len();
        data.into_iter().for_each(|x| {
            if x.len() != ncol { panic!("inconstent row lengths") }
            total.extend(x); });

        Self(T::from_flat((nrow, ncol).into(), total))
    }

    pub fn from_points_row(points: Vec<Point>) -> Self {
        // construct a matrix from a series of points as row vectors
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let dim = points[0].dim();

        for p in points.into_iter() {
            if p.dim() != dim { panic!("inconsistent point dims") }
            rows.push(p.into());
        }

        Self::from_rows(rows)
    }

    pub fn transpose(&mut self) {
        self.0.transpose();
    }

    pub fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }

    pub fn row(&self, r: usize) -> MatrixRow<T> {
        self.0.row(r)
    }

    pub fn col(&self, c: usize) -> MatrixCol<T> {
        self.0.col(c)
    }

    pub fn swap_rows(&mut self, i: usize, j: usize) {
        self.0.swap_rows(i, j);
    }

    pub fn swap_cols(&mut self, i: usize, j: usize) {
        self.0.swap_cols(i, j);
    }

    pub fn solve_gausselim(&mut self, b: Matrix<T>) -> Result<Matrix<T>, String> {
        let res = self.0.solve_gausselim(b.0);
        if let Ok(x) = res { return Ok(Matrix(x)) }
        match res {
            Ok(x) => Ok(Matrix(x)),
            Err(e) => Err(e),
        }
    }

    pub fn frobenius(&self) -> f64 {
        self.0.frobenius()
    }
}

impl LinearMatrix {
    pub fn new(buff: buffer::LinearBuffer) -> Self {
        Matrix(buff)
    }

    pub fn from_flat(shape: MatrixShape, data: Vec<f64>) -> Self {
        // eventually, this will be clever about what kind of buffer it chooses
        LinearMatrix::new(buffer::LinearBuffer::from_flat(shape.into(), data))
    }

    pub fn eye(dim: usize) -> Self {
        LinearMatrix::new(buffer::LinearBuffer::eye(dim))
    }

    pub fn zeros(shape: MatrixShape) -> Self {
        LinearMatrix::new(buffer::LinearBuffer::zeros(shape))
    }
}

impl<'a, T> Add<&'a Matrix<T>> for &'a Matrix<T>
    where T: MatrixData
{
    type Output = Matrix<T>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut new = self.0.clone();
        new.add(&rhs.0);
        Matrix(new)
    }
}

impl<'a, T> Sub<&'a Matrix<T>> for &'a Matrix<T> 
    where T: MatrixData
{
    type Output = Matrix<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut new = self.0.clone();
        new.sub(&rhs.0);
        Matrix(new)
    }
}

impl<'a, T> Mul<&'a Matrix<T>> for &'a Matrix<T>
    where T: MatrixData
{
    type Output = Matrix<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        Matrix(self.0.mul(&rhs.0))
    }
}

impl<T> PartialEq<Matrix<T>> for Matrix<T>
    where T: MatrixData + PartialEq
{
    fn eq(&self, rhs: &Self) -> bool {
        &self.0 == &rhs.0
    }
}