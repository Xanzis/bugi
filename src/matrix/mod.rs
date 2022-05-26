use crate::spatial::Point;
use std::cmp::max;
use std::error;
use std::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul};

pub mod buffer;
pub mod graph;
pub mod inverse;
pub mod norm;
pub mod solve;
pub mod sparse;
pub use inverse::Inverse;
pub use norm::Norm;

#[cfg(test)]
mod tests;

pub use buffer::{ConstMatrix, Diagonal, LinearMatrix, LowerTriangular, UpperTriangular};
pub use sparse::{BiEnvelope, CompressedRow, Dictionary, LowerRowEnvelope};
//type Matrix = LinearMatrix; // default implementation

// rolling my own (pretty limited) matrix math
// standard order for shape is (row, col)

#[derive(Debug, Clone)]
pub enum MatrixError {
    SolveError(String),
    Pivot,
}

impl MatrixError {
    fn solve<T: ToString>(msg: T) -> Self {
        MatrixError::SolveError(msg.to_string())
    }
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::SolveError(s) => write!(f, "SolveError: {}", s),
            MatrixError::Pivot => write!(f, "Pivot failed, matrix may be singular"),
        }
    }
}

impl error::Error for MatrixError {}

// data type for matrix shapes
// will eventually add fields to help initialize sparse matrices
#[derive(Clone, Copy, Debug)]
pub struct MatrixShape {
    ncol: usize,
    nrow: usize,
    max_dim: usize,
}

impl MatrixShape {
    pub fn to_rc(self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }
}

impl From<(usize, usize)> for MatrixShape {
    fn from(dims: (usize, usize)) -> Self {
        MatrixShape {
            nrow: dims.0,
            ncol: dims.1,
            max_dim: max(dims.0, dims.1),
        }
    }
}

impl From<usize> for MatrixShape {
    fn from(dim: usize) -> Self {
        MatrixShape {
            nrow: dim,
            ncol: dim,
            max_dim: dim,
        }
    }
}

pub trait MatrixLike
where
    Self: Sized + Clone + fmt::Debug + fmt::Display,
    Self: Index<(usize, usize), Output = f64> + IndexMut<(usize, usize), Output = f64>,
{
    // basic required methods
    fn shape(&self) -> (usize, usize);
    fn get(&self, loc: (usize, usize)) -> Option<&f64>;
    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64>;
    fn transpose(&mut self);
    fn zeros<T: Into<MatrixShape>>(shape: T) -> Self;
    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(shape: T, data: U) -> Self;

    // provided methods

    fn put(&mut self, loc: (usize, usize), val: f64) {
        if let Some(x) = self.get_mut(loc) {
            *x = val;
        }
    }

    fn mutate<F>(&mut self, loc: (usize, usize), mut f: F)
    where
        F: FnMut(&mut f64),
    {
        if let Some(x) = self.get_mut(loc) {
            f(x);
        }
    }

    fn add_ass(&mut self, other: &Self) {
        if self.shape() != other.shape() {
            panic!("bad shapes: {:?}, {:?}", self.shape(), other.shape());
        }
        for r in 0..self.shape().0 {
            for c in 0..self.shape().1 {
                self[(r, c)] += other[(r, c)];
            }
        }
    }

    fn sub_ass(&mut self, other: &Self) {
        if self.shape() != other.shape() {
            panic!("bad shapes: {:?}, {:?}", self.shape(), other.shape());
        }
        for r in 0..self.shape().0 {
            for c in 0..self.shape().1 {
                self[(r, c)] -= other[(r, c)];
            }
        }
    }

    fn mul<T: MatrixLike, U: MatrixLike>(&self, other: &T) -> U {
        // generics are nice but this will be inefficient for sparse / triangular
        let a_shape = self.shape();
        let b_shape = other.shape();

        if a_shape.1 != b_shape.0 {
            panic!(
                "improper shapes for matrix multiplication: {:?} and {:?}",
                a_shape, b_shape
            )
        }

        let res_shape = (a_shape.0, b_shape.1);
        let mut res_vals = Vec::with_capacity(res_shape.0 * res_shape.1);

        for r in 0..res_shape.0 {
            for c in 0..res_shape.1 {
                res_vals.push(dot(self.row(r), other.col(c)));
            }
        }
        U::from_flat(res_shape, res_vals)
    }

    // *****
    // methods for manipulating rows/columns

    fn row(&self, i: usize) -> MatrixRow<Self> {
        if i < self.shape().0 {
            return MatrixRow {
                source: &self,
                row: i,
                pos: 0,
            };
        } else {
            panic!("index out of bounds")
        }
    }

    fn col(&self, i: usize) -> MatrixCol<Self> {
        if i < self.shape().1 {
            return MatrixCol {
                source: &self,
                col: i,
                pos: 0,
            };
        } else {
            panic!("index out of bounds")
        }
    }

    fn diag(&self) -> MatrixDiag<Self> {
        MatrixDiag {
            source: &self,
            pos: 0,
        }
    }

    fn flat(&self) -> MatrixAll<Self> {
        // visit all of the matrix elements in row-major order
        MatrixAll::new(&self)
    }

    fn indices(&self) -> MatrixIndices {
        // visit all valid matrix indices in row-major order
        MatrixIndices::new(self)
    }

    fn set_row(&mut self, i: usize, new: Vec<f64>) {
        if new.len() != self.shape().1 {
            panic!("incompatible row length")
        }
        for (c, val) in new.into_iter().enumerate() {
            self[(i, c)] = val;
        }
    }

    fn set_col(&mut self, i: usize, new: Vec<f64>) {
        if new.len() != self.shape().0 {
            panic!("incompatible column length")
        }
        for (r, val) in new.into_iter().enumerate() {
            self[(r, i)] = val;
        }
    }

    fn swap_rows(&mut self, i: usize, j: usize) {
        for c in 0..self.shape().1 {
            let temp = self[(i, c)];
            self[(i, c)] = self[(j, c)];
            self[(j, c)] = temp;
        }
    }

    fn swap_cols(&mut self, i: usize, j: usize) {
        for r in 0..self.shape().0 {
            let temp = self[(r, i)];
            self[(r, i)] = self[(r, j)];
            self[(r, j)] = temp;
        }
    }

    fn mutate_row<F>(&mut self, i: usize, mut f: F)
    where
        F: FnMut(&mut f64),
    {
        for c in 0..self.shape().1 {
            f(self.get_mut((i, c)).unwrap());
        }
    }

    fn mutate_col<F>(&mut self, i: usize, mut f: F)
    where
        F: FnMut(&mut f64),
    {
        for r in 0..self.shape().0 {
            f(self.get_mut((r, i)).unwrap());
        }
    }

    // *****
    // matrix initialization methods

    fn eye<T: Into<MatrixShape>>(dim: T) -> Self {
        let shape = dim.into();
        let mut res = Self::zeros(shape.max_dim);
        // this is the naive way - this can certainly be done faster
        for i in 0..shape.max_dim {
            res.put((i, i), 1.0);
        }
        res
    }

    fn from_rows(data: Vec<Vec<f64>>) -> Self {
        let mut total = Vec::new();
        let nrow = data.len();
        if nrow == 0 {
            panic!("empty data")
        }
        let ncol = data.get(0).expect("empty first row").len();
        for x in data.into_iter() {
            if x.len() != ncol {
                panic!("inconsistent row lengths");
            }
            total.extend(x);
        }

        Self::from_flat((nrow, ncol), total)
    }

    fn from_points_row(points: Vec<Point>) -> Self {
        // construct a matrix from a series of points as row vectors
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let dim = points[0].dim();

        for p in points.into_iter() {
            if p.dim() != dim {
                panic!("inconsistent point dims")
            }
            rows.push(p.into());
        }

        Self::from_rows(rows)
    }

    fn row_vec(data: Vec<f64>) -> Self {
        // construct a (1, n) matrix
        Self::from_flat((1, data.len()), data)
    }

    fn col_vec(data: Vec<f64>) -> Self {
        let mut res = Self::row_vec(data);
        res.transpose();
        res
    }

    // *****
    // matrix display methods (specialization instability prevents generic Display)

    fn disp(&self) -> String {
        let (nrow, ncol) = self.shape();
        let mut res = format!("rows: {} cols: {}\n", nrow, ncol);
        for i in 0..nrow {
            for val in self.row(i) {
                let new = format!("{:>+07.1e} ", val);
                res.push_str(new.as_str());
            }
            res.push_str("\n");
        }
        res
    }
}

pub struct MatrixRow<'a, T>
where
    T: MatrixLike,
{
    source: &'a T,
    row: usize,
    pos: usize,
}

pub struct MatrixCol<'a, T>
where
    T: MatrixLike,
{
    source: &'a T,
    col: usize,
    pos: usize,
}

pub struct MatrixDiag<'a, T>
where
    T: MatrixLike,
{
    source: &'a T,
    pos: usize,
}

pub struct MatrixAll<'a, T>
where
    T: MatrixLike,
{
    source: &'a T,
    row: usize,
    col: usize,
}

pub struct MatrixIndices {
    rows: usize,
    cols: usize,
    r: usize,
    c: usize,
}

impl<'a, T> Iterator for MatrixCol<'a, T>
where
    T: MatrixLike,
{
    type Item = &'a f64;

    fn next(&mut self) -> Option<&'a f64> {
        let loc = self.pos;
        self.pos += 1;
        self.source.get((loc, self.col))
    }
}

impl<'a, T> Iterator for MatrixRow<'a, T>
where
    T: MatrixLike,
{
    type Item = &'a f64;

    fn next(&mut self) -> Option<&'a f64> {
        let loc = self.pos;
        self.pos += 1;
        self.source.get((self.row, loc))
    }
}

impl<'a, T> Iterator for MatrixDiag<'a, T>
where
    T: MatrixLike,
{
    type Item = &'a f64;

    fn next(&mut self) -> Option<&'a f64> {
        let loc = self.pos;
        self.pos += 1;
        self.source.get((loc, loc))
    }
}

impl<'a, T> Iterator for MatrixAll<'a, T>
where
    T: MatrixLike,
{
    type Item = &'a f64;

    fn next(&mut self) -> Option<&'a f64> {
        let x = self.source.get((self.row, self.col));
        self.col += 1;
        if self.col >= self.source.shape().1 {
            self.col = 0;
            self.row += 1;
        }
        x
    }
}

impl Iterator for MatrixIndices {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.r >= self.rows {
            return None;
        }

        let res = (self.r, self.c);

        self.c += 1;
        if self.c >= self.cols {
            self.c = 0;
            self.r += 1;
        }

        Some(res)
    }
}

impl<'a, T> MatrixAll<'a, T>
where
    T: MatrixLike,
{
    fn new(source: &'a T) -> Self {
        MatrixAll {
            source,
            row: 0,
            col: 0,
        }
    }
}

impl MatrixIndices {
    fn new<T: MatrixLike>(source: &T) -> Self {
        let (rows, cols) = source.shape();
        MatrixIndices {
            rows,
            cols,
            r: 0,
            c: 0,
        }
    }
}

// miscellaneous math functionality

#[derive(Clone, Debug)]
pub struct Average<T>
//    where T: Clone + Div<f64> + Mul<f64> + Add
where
    T: Div<f64, Output = T> + Mul<f64, Output = T> + Add<Output = T> + Clone,
{
    n: usize,
    avg: Option<T>,
}

impl<T> Average<T>
where
    T: Div<f64, Output = T> + Mul<f64, Output = T> + Add<Output = T> + Clone,
{
    pub fn new() -> Self {
        Self { n: 0, avg: None }
    }

    pub fn update(&mut self, val: T) {
        if let Some(avg) = self.avg.clone() {
            let temp = avg * (self.n as f64) + val;
            self.n += 1;
            self.avg = Some(temp / (self.n as f64));
        } else {
            self.n = 1;
            self.avg = Some(val);
        }
    }

    pub fn consume(self) -> Option<T> {
        self.avg
    }
}

#[inline]
pub fn dot<'a, T, U>(x: T, y: U) -> f64
where
    T: IntoIterator<Item = &'a f64>,
    U: IntoIterator<Item = &'a f64>,
{
    x.into_iter().zip(y.into_iter()).map(|(a, b)| a * b).sum()
}
