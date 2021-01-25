use crate::spatial::Point;
use std::fmt;
use std::cmp::PartialEq;
use std::ops::{Add, Mul, Sub};

pub mod inverse;
pub mod norm;
pub mod buffer;

// rolling my own (pretty limited) matrix math
// standard order for shape is (row, col)
// m[r][c] is stored at data[r*ncol + c]
type Matrix = Box<dyn MatrixData>

// data type for matrix sizes
// will eventually add fields to help initialize sparse matrices
pub struct MatrixShape {
    ncol: usize,
    nrow: usize,
}

pub struct MatrixRow<'a, T>
    where T: MatrixData
{
    source: &'a T,
    row: usize,
    pos: usize,
}

pub struct MatrixCol<'a, T>
    where T: MatrixData
{
    source: &'a T,
    col: usize,
    pos: usize,
}

impl fmt::Display for Matrix
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mat_data = *self;
        let (nrow, ncol) = mat_data.shape();
        write!(f, "rows: {} cols: {}\n", nrow, ncol)?;
        for i in 0..nrow {
            for j in 0..ncol {
                write!(f, "{:1.5} ", data.get((i, j)).unwrap())?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

pub trait MatrixData {
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
        f(self.get_mut(loc).unwrap());
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
}

impl<'a, T> Iterator for MatrixCol<'a, T>
    where T: MatrixData
{
    type Item = &'a f64;

    fn next(&mut self) -> Option<&'a f64> {
        let loc = self.pos;
        self.pos += 1;
        self.source.get((loc, self.col))
    }
}

impl<'a, T> Iterator for MatrixRow<'a, T>
    where T: MatrixData
{
    type Item = &'a f64;

    fn next(&mut self) -> Option<&'a f64> {
        let loc = self.pos;
        self.pos += 1;
        self.source.get((self.row, loc))
    }
}

impl From<(usize, usize)> for MatrixShape {
    fn from(dims: (usize, usize)) -> Self {
        MatrixShape{ nrow: dims.0, ncol: dims.1 }
    }
}

impl<T: MatrixData> Matrix<T> {

    pub fn from_rows(data: Vec<Vec<f64>>) -> Self {
        let mut total = Vec::new();
        let nrow = data.len();
        if nrow == 0 { panic!("empty data") }
        let ncol = data.get(0).unwrap().len();
        data.iter().foreach(|x| {
            if x.len() != ncol { panic!("inconstent row lengths") }
            total.append(x); });

        Self::from_flat((nrow, ncol).into(), total)
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
}

impl Matrix<buffer::LinearBuffer> {
    pub fn from_flat(shape: MatrixShape, data: Vec<f64>) -> Self {
        // eventually, this will be clever about what kind of buffer it chooses
        Matrix(buffer::LinearBuffer::from_flat(shape.into(), data))
    }

    pub fn eye(dim: usize) -> Self {
        Matrix(buffer::LinearBuffer::eye(dim))
    }
}

impl<'a, T> Add<&'a Matrix<T>> for &'a Matrix<T> 
    where T: Add + MatrixData
{
    type Output = Matrix<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Matrix(&self.0 + &rhs.0)
    }
}

impl<'a, T> Sub<&'a Matrix<T>> for &'a Matrix<T>
    where T: Sub + MatrixData
{
    type Output = Matrix<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Matrix(&self.0 - &rhs.0)
    }
}

impl<'a, T> Mul<&'a Matrix<T>> for &'a Matrix<T>
    where T: Mul + MatrixData
{
    type Output = Matrix<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        Matrix(&self.0 * &rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;
    #[test]
    fn zeros() {
        let target = Matrix::from_flat((3, 2).into(), vec![0.0; 6]);
        assert_eq!(target, Matrix::zeros((3, 2)));
    }
    #[test]
    fn eye() {
        let target = Matrix::from_flat((2, 2).into(), vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(target, Matrix::eye(2));
    }
    #[test]
    fn matadd() {
        let a = rm_from_flat((2, 3).into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0]);
        let b = Matrix::from_rows(vec![vec![4.0, 5.0, 6.0], vec![1.0, 2.0, 3.0]]);
        let sum = &a + &b;
        let target = Matrix::from_rows(vec![vec![5.0, 7.0, 9.0], vec![5.0, 7.0, 10.0]]);
        assert_eq!(sum, target);
    }
    #[test]
    fn matmul() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 7.0]]);
        let b = Matrix::from_rows(vec![
            vec![4.0, 2.0, 2.0, 1.0],
            vec![1.0, 2.0, 1.0, 1.0],
            vec![1.0, 2.0, 1.0, 2.0],
        ]);
        let target = Matrix::from_rows(vec![
            vec![9.0, 12.0, 7.0, 9.0],
            vec![28.0, 32.0, 20.0, 23.0],
        ]);
        let prod = &a * &b;
        assert_eq!(target, prod);
    }
    #[test]
    fn disp() {
        let target = "rows: 1 cols: 2\n1.00000 2.00000 \n";
        let mat = Matrix::from_rows(vec![vec![1.0, 2.0]]);
        println!("{}", mat);
        assert_eq!(format!("{}", mat), target)
    }
    #[test]
    fn transpose() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let a = a.transpose();
        let b = Matrix::from_rows(vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
        assert_eq!(a, b);
    }
    #[test]
    fn swaps() {
        let mut a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let target_a = Matrix::from_rows(vec![vec![3.0, 4.0], vec![1.0, 2.0]]);
        a.swap_rows(0, 1);
        assert_eq!(a, target_a);
        let target_b = Matrix::from_rows(vec![vec![4.0, 3.0], vec![2.0, 1.0]]);
        a.swap_cols(0, 1);
        assert_eq!(a, target_b);
    }
    #[test]
    fn solve_gauss() {
        let mut a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_rows(vec![vec![5.0], vec![6.0]]);

        let x = a.solve_gausselim(b).unwrap();

        let target_x = Matrix::from_rows(vec![vec![-4.0], vec![4.5]]);
        assert!((&x - &target_x).frobenius() < 1e-10);
    }
}
