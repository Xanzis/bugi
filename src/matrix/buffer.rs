use super::{MatrixShape, MatrixLike};
use std::cmp::PartialEq;
use std::ops::{Add, Sub, AddAssign, MulAssign, Index, IndexMut};
use std::fmt;

#[derive(Clone, Debug)]
pub struct LinearMatrix {
    dims: (usize, usize),
    row_maj: bool,
    data: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct LowerTriangular {
    dim: usize,
    data: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct UpperTriangular {
    dim: usize,
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

impl LowerTriangular {
    fn pos(&self, loc: (usize, usize)) -> Option<usize> {
        let (row, col) = loc;
        if (row >= self.dim) || (col >= self.dim) { return None }
        // returns None if a non-triangular position is requested
        // Index implementation should fill &0.0
        // IndexMut should panic
        if col > row { return None }
        Some((row * (row + 1) / 2) + col)
    }

    pub fn forward_sub(&self, b: &[f64]) -> Vec<f64> {
        // solves Lx = b
        // TODO consider returning a Result
        println!("this is me: {}", self);
        let dim = self.dim;
        if dim != b.len() {panic!("incompatible shapes")}
        let mut x = vec![0.0; dim];
        for i in 0..dim {
            let mut temp = 0.0;
            for j in 0..i {
                temp += self[(i, j)] * x[j];
            }
            x[i] = (b[i] - temp) / self[(i, i)];
            println!("assigned x[{}] = {}", i, x[i]);
        }
        x
    }

    pub fn tri_inv(&self) -> Self {
        // solves LX = I
        let mut b = vec![0.0; self.dim];
        let mut res = LowerTriangular::zeros(self.dim);
        for i in 0..self.dim {
            // invert identity vectors column-by-column
            b[i] = 1.0;
            let x = self.forward_sub(b.as_slice());
            // x will begin with i 0s (since X is lower triangular)
            for (j, val) in x.into_iter().enumerate().skip(i) {
                res[(j, i)] = val;
            }
            b[i] = 0.0; // put it back :)
        }
        res
    }
}

impl UpperTriangular {
    fn pos(&self, loc: (usize, usize)) -> Option<usize> {
        let (row, col) = loc;
        if (col >= self.dim) || (row >= self.dim) { return None }
        if row > col { return None }
        Some((col * (col + 1) / 2) + row)
    }

    pub fn backward_sub(&self, b: &[f64]) -> Vec<f64> {
        // solves Ux = b
        let dim = self.dim;
        if dim != b.len() {panic!("incompatible shapes")}
        let mut x = vec![0.0; dim];
        for i in (0..dim).rev() {
            let mut temp = 0.0;
            for j in (i + 1)..dim {
                temp += self[(i, j)] * x[j];
            }
            println!("{:?}", temp);
            x[i] = (b[i] - temp) / self[(i, i)];
        }
        x
    }

    pub fn tri_inv(&self) -> Self {
        // solves UX = I
        let mut b = vec![0.0; self.dim];
        let mut res = UpperTriangular::zeros(self.dim);
        for i in 0..self.dim {
            // invert identity vectors column-by-column
            b[i] = 1.0;
            let x = self.backward_sub(b.as_slice());
            // x's first (i + 1) values will be nonzero
            for (j, val) in x.into_iter().enumerate().take(i + 1) {
                res[(j, i)] = val;
            }
            b[i] = 0.0; // put it back :)
        }
        res
    }
}

// *****
// LinearMatrix Implementations
// *****

impl MatrixLike for LinearMatrix {
    fn shape(&self) -> (usize, usize) {
        self.dims
    }

    fn get(&self, loc: (usize, usize)) -> Option<&f64> {
        if let Some(i) = self.pos(loc) {
            // this is fine, self.pos checks bounds
            unsafe {Some(self.data.get_unchecked(i))}
        }
        else { None }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        if let Some(i) = self.pos(loc) {
            unsafe {Some(self.data.get_unchecked_mut(i))}
        }
        else { None }
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

impl Index<(usize, usize)> for LinearMatrix {
    type Output = f64;
    fn index(&self, loc: (usize, usize)) -> &Self::Output {
        if let Some(i) = self.pos(loc) {
            // this is perfectly fine, self.pos performs bounds check
            unsafe {
                self.data.get_unchecked(i)
            }
        }
        else {
            panic!("matrix index out of bounds")
        }
    }
}

impl IndexMut<(usize, usize)> for LinearMatrix {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        if let Some(i) = self.pos(loc) {
            unsafe {
                self.data.get_unchecked_mut(i)
            }
        }
        else {
            panic!("matrix index out of bounds")
        }
    }
}

// *****
// LowerTriangular Implementations
// *****

impl MatrixLike for LowerTriangular {
    fn shape(&self) -> (usize, usize) {
        (self.dim, self.dim)
    }

    fn get(&self, loc: (usize, usize)) -> Option<&f64> {
        if let Some(i) = self.pos(loc) {
            // this is fine, self.pos checks bounds
            unsafe {Some(self.data.get_unchecked(i))}
        }
        else if (loc.0 < self.dim) && (loc.1 < self.dim) {
            Some(&0.0)
        }
        else { None }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        if let Some(i) = self.pos(loc) {
            unsafe {Some(self.data.get_unchecked_mut(i))}
        }
        else { None }
    }

    fn transpose(&mut self) {
        unimplemented!() // cannot transpose this type in place
    }

    fn zeros<T: Into<MatrixShape>>(shape: T) -> Self {
        let shape: MatrixShape = shape.into();
        let dim = shape.max_dim;
        let data = vec![0.0; (dim + 1) * (dim + 2) / 2];
        Self { dim, data }
    }

    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(shape: T, data: U) -> Self {
        // don't feel like it
        unimplemented!()
    }
}

impl PartialEq for LowerTriangular {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl fmt::Display for LowerTriangular {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.disp())
    }
}

impl Index<(usize, usize)> for LowerTriangular {
    type Output = f64;
    fn index(&self, loc: (usize, usize)) -> &Self::Output {
        if let Some(i) = self.pos(loc) {
            // this is perfectly fine, self.pos performs bounds check
            unsafe {self.data.get_unchecked(i)}
        }
        else if (loc.0 < self.dim) && (loc.1 < self.dim) {
            &0.0
        }
        else {panic!("matrix index out of bounds")}
    }
}

impl IndexMut<(usize, usize)> for LowerTriangular {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        if let Some(i) = self.pos(loc) {
            unsafe {self.data.get_unchecked_mut(i)}
        }
        else {panic!("matrix index out of bounds")}
    }
}

// *****
// UpperTriangular Implementations
// *****

impl MatrixLike for UpperTriangular {
    fn shape(&self) -> (usize, usize) {
        (self.dim, self.dim)
    }

    fn get(&self, loc: (usize, usize)) -> Option<&f64> {
        if let Some(i) = self.pos(loc) {
            // this is fine, self.pos checks bounds
            unsafe {Some(self.data.get_unchecked(i))}
        }
        else if (loc.0 < self.dim) && (loc.1 < self.dim) {
            Some(&0.0)
        }
        else { None }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        if let Some(i) = self.pos(loc) {
            unsafe {Some(self.data.get_unchecked_mut(i))}
        }
        else { None }
    }

    fn transpose(&mut self) {
        unimplemented!() // cannot transpose this type in place
    }

    fn zeros<T: Into<MatrixShape>>(shape: T) -> Self {
        let shape: MatrixShape = shape.into();
        let dim = shape.max_dim;
        let data = vec![0.0; (dim + 1) * (dim + 2) / 2];
        Self { dim, data }
    }

    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(shape: T, data: U) -> Self {
        unimplemented!()
    }
}

impl PartialEq for UpperTriangular {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl fmt::Display for UpperTriangular {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.disp())
    }
}

impl Index<(usize, usize)> for UpperTriangular {
    type Output = f64;
    fn index(&self, loc: (usize, usize)) -> &Self::Output {
        if let Some(i) = self.pos(loc) {
            // this is perfectly fine, self.pos performs bounds check
            unsafe {self.data.get_unchecked(i)}
        }
        else if (loc.1 < self.dim) && (loc.0 < self.dim) {
            &0.0
        }
        else {panic!("matrix index out of bounds")}
    }
}

impl IndexMut<(usize, usize)> for UpperTriangular {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        if let Some(i) = self.pos(loc) {
            unsafe {self.data.get_unchecked_mut(i)}
        }
        else {panic!("matrix index out of bounds")}
    }
}