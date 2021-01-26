use crate::spatial::Point;
use super::{MatrixShape};
use std::cmp::PartialEq;
use std::ops::{Add, Mul, Sub};
use std::fmt;

pub trait MatrixLike
    where Self: Sized + Clone + fmt::Debug + fmt::Display
{
    // basic required methods
    fn shape(&self) -> (usize, usize);
    fn get(&self, loc: (usize, usize)) -> Option<&f64>;
    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64>;
    fn transpose(&mut self);
    fn zeros(shape: MatrixShape) -> Self;
    fn from_flat(shape: MatrixShape, data: Vec<f64>) -> Self;

    // provided methods

    fn put(&mut self, loc:(usize, usize), val: f64) {
        if let Some(x) = self.get_mut(loc) {
            *x = val;
        }
    }

    fn mutate<F>(&mut self, loc: (usize, usize), mut f: F)
    where
        F: FnMut(&mut f64),
    {
        if let Some(x) = self.get_mut(loc) { f(x); }
    }

    fn add_ass(&mut self, other: &Self) {
        if self.shape() != other.shape() {
            panic!("bad shapes: {:?}, {:?}", self.shape(), other.shape());
        }
        for r in 0..self.shape().0 {
            for c in 0..self.shape().1 {
                let y = other.get((r, c)).unwrap();
                self.mutate((r, c), |x| *x += y);
            }
        }
    }

    fn sub_ass(&mut self, other: &Self) {
        if self.shape() != other.shape() {
            panic!("bad shapes: {:?}, {:?}", self.shape(), other.shape());
        }
        for r in 0..self.shape().0 {
            for c in 0..self.shape().1 {
                let y = other.get((r, c)).unwrap();
                self.mutate((r, c), |x| *x -= y);
            }
        }
    }

    fn mul(&self, other: &Self) -> Self {
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
                let dot = self.row(r)
                    .zip(other.col(c))
                    .map(|x| x.0 * x.1).sum();
                res_vals.push(dot);
            }
        }
        Self::from_flat(res_shape.into(), res_vals)
    }

    // *****
    // methods for manipulating rows/columns

    fn row(&self, i: usize) -> MatrixRow<Self> {
        if i < self.shape().0 {
            return MatrixRow { source: &self, row: i, pos: 0 };
        }
        else { panic!("index out of bounds") }
    }

    fn col(&self, i: usize) -> MatrixCol<Self> {
        if i < self.shape().1 {
            return MatrixCol { source: &self, col: i, pos: 0 };
        }
        else { panic!("index out of bounds") }
    }

    fn flat(&self) -> MatrixAll<Self> {
        // visit all of the matrix elements in row-major order
        MatrixAll::new(&self)
    }

    fn set_row(&mut self, i: usize, new: Vec<f64>) {
        if new.len() != self.shape().1 { panic!("incompatible row length") }
        for (c, val) in new.into_iter().enumerate() {
            self.put((i, c), val);
        }
    }

    fn set_col(&mut self, i: usize, new: Vec<f64>) {
        if new.len() != self.shape().0 { panic!("incompatible column length") }
        for (r, val) in new.into_iter().enumerate() {
            self.put((r, i), val);
        }
    }

    fn swap_rows(&mut self, i: usize, j: usize) {
        let temp: Vec<f64> = self.row(j).cloned().collect();
        self.set_row(j, self.row(i).cloned().collect());
        self.set_row(i, temp);
    }

    fn swap_cols(&mut self, i: usize, j: usize) {
        let temp: Vec<f64> = self.col(j).cloned().collect();
        self.set_col(j, self.col(i).cloned().collect());
        self.set_col(i, temp);
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

    fn eye(dim: usize) -> Self {
        let mut res = Self::zeros((dim, dim).into());
        // this is the naive way - this can certainly be done faster
        for i in 0..dim { res.put((i, i), 1.0); }
        res
    }

    fn from_rows(data: Vec<Vec<f64>>) -> Self {
        let mut total = Vec::new();
        let nrow = data.len();
        if nrow == 0 { panic!("empty data") }
        let ncol = data.get(0).unwrap().len();
        data.into_iter().for_each(|x| {
            if x.len() != ncol { panic!("inconstent row lengths") }
            total.extend(x); });

        Self::from_flat((nrow, ncol).into(), total)
    }

    fn from_points_row(points: Vec<Point>) -> Self {
        // construct a matrix from a series of points as row vectors
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let dim = points[0].dim();

        for p in points.into_iter() {
            if p.dim() != dim { panic!("inconsistent point dims") }
            rows.push(p.into());
        }

        Self::from_rows(rows)
    }
}

pub struct MatrixRow<'a, T>
    where T: MatrixLike
{
    source: &'a T,
    row: usize,
    pos: usize,
}

pub struct MatrixCol<'a, T>
    where T: MatrixLike
{
    source: &'a T,
    col: usize,
    pos: usize,
}

pub struct MatrixAll<'a, T>
    where T: MatrixLike
{
    source: &'a T,
    row_pos: usize,
    col_pos: usize,
}

impl<'a, T> Iterator for MatrixCol<'a, T>
    where T: MatrixLike
{
    type Item = &'a f64;

    fn next(&mut self) -> Option<&'a f64> {
        let loc = self.pos;
        self.pos += 1;
        self.source.get((loc, self.col))
    }
}

impl<'a, T> Iterator for MatrixRow<'a, T>
    where T: MatrixLike
{
    type Item = &'a f64;

    fn next(&mut self) -> Option<&'a f64> {
        let loc = self.pos;
        self.pos += 1;
        self.source.get((self.row, loc))
    }
}

impl<'a, T> Iterator for MatrixAll<'a, T>
    where T: MatrixLike
{
    type Item = &'a f64;

    fn next(&mut self) -> Option<&'a f64> {
        let x = self.source.get((self.row_pos, self.col_pos));
        self.row_pos += 1;
        if self.row_pos >= self.source.shape().1 {
            self.row_pos = 0;
            self.col_pos += 1;
        }
        x
    }
}

impl<'a, T> MatrixAll<'a, T>
    where T: MatrixLike
{
    fn new(source: &'a T) -> Self {
        MatrixAll{ source, row_pos: 0, col_pos: 0 }
    }
}

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
            Some((self.dims.0 * loc.0) + loc.1)
        }
        else {
            Some((self.dims.1 * loc.1) + loc.0)
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
    }

    fn zeros(shape: MatrixShape) -> Self {
        let dims = (shape.nrow, shape.ncol);
        let data = vec![0.0; dims.0 * dims.1];
        Self { dims, row_maj: true, data }
    }
    fn from_flat(shape: MatrixShape, data: Vec<f64>) -> Self {
        // turn a row-major vector of values into a matrix
        let dims = (shape.nrow, shape.ncol);
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