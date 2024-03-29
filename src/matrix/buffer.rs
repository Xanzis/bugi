use super::{MatrixLike, MatrixShape};
use std::cmp::PartialEq;
use std::fmt;
use std::ops::{Add, AddAssign, Index, IndexMut, MulAssign, Sub};

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

#[derive(Clone, Debug)]
pub struct Diagonal {
    data: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstMatrix<const SIZE: usize> {
    dims: (usize, usize),
    data: [f64; SIZE],
}

impl LinearMatrix {
    fn pos(&self, loc: (usize, usize)) -> Option<usize> {
        // find the location of the desired element in data
        // read differently for row/column major orders of buffer
        if (loc.1 >= self.dims.1) || (loc.0 >= self.dims.0) {
            return None;
        }
        if self.row_maj {
            Some((self.dims.1 * loc.0) + loc.1)
        } else {
            Some((self.dims.0 * loc.1) + loc.0)
        }
    }
}

impl LowerTriangular {
    fn pos(&self, loc: (usize, usize)) -> Option<usize> {
        let (row, col) = loc;
        if (row >= self.dim) || (col >= self.dim) {
            return None;
        }
        // returns None if a non-triangular position is requested
        // Index implementation should fill &0.0
        // IndexMut should panic
        if col > row {
            return None;
        }
        Some((row * (row + 1) / 2) + col)
    }

    pub fn forward_sub(&self, b: &[f64]) -> Vec<f64> {
        // solves Lx = b
        let dim = self.dim;
        if dim != b.len() {
            panic!("incompatible shapes")
        }
        let mut x = vec![0.0; dim];
        for i in 0..dim {
            let temp: f64 = self.row(i).zip(x.iter()).map(|(x, y)| x * y).take(i).sum();
            x[i] = (b[i] - temp) / self[(i, i)];
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
        if (col >= self.dim) || (row >= self.dim) {
            return None;
        }
        if row > col {
            return None;
        }
        Some((col * (col + 1) / 2) + row)
    }

    pub fn backward_sub(&self, b: &[f64]) -> Vec<f64> {
        // solves Ux = b
        let dim = self.dim;
        if dim != b.len() {
            panic!("incompatible shapes")
        }
        let mut x = vec![0.0; dim];
        for i in (0..dim).rev() {
            let mut temp = 0.0;
            for j in (i + 1)..dim {
                temp += self[(i, j)] * x[j];
            }
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

impl Diagonal {
    pub fn num_neg(&self) -> usize {
        // count the number of negative entries
        self.data.iter().filter(|&&x| x < 0.0).count()
    }

    pub fn product(&self) -> f64 {
        self.data.iter().product()
    }

    pub fn product_scaled(&self, scale: f64) -> f64 {
        self.data.iter().map(|x| x * scale).product()
    }

    pub fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / (self.data.len() as f64)
    }

    pub fn solve(&self, x: &mut [f64]) {
        // solve Dx = x in-place
        assert_eq!(x.len(), self.data.len());

        for i in 0..self.data.len() {
            x[i] /= self.data[i];
        }
    }
}

impl<const SIZE: usize> ConstMatrix<SIZE> {
    fn pos(&self, loc: (usize, usize)) -> Option<usize> {
        if (loc.1 >= self.dims.1) || (loc.0 >= self.dims.0) {
            return None;
        }
        Some((self.dims.1 * loc.0) + loc.1)
    }

    pub fn from_flat_array<T: Into<MatrixShape>>(shape: T, data: [f64; SIZE]) -> Self {
        let shape = shape.into();
        let dims = (shape.nrow, shape.ncol);

        assert_eq!(dims.0 * dims.1, SIZE, "provided array does not match shape");

        Self { dims, data }
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
            unsafe { Some(self.data.get_unchecked(i)) }
        } else {
            None
        }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        if let Some(i) = self.pos(loc) {
            unsafe { Some(self.data.get_unchecked_mut(i)) }
        } else {
            None
        }
    }

    fn transpose(&mut self) {
        self.row_maj = !self.row_maj;
        self.dims = (self.dims.1, self.dims.0);
    }

    fn zeros<T: Into<MatrixShape>>(shape: T) -> Self {
        let shape: MatrixShape = shape.into();
        let dims = (shape.nrow, shape.ncol);
        let data = vec![0.0; dims.0 * dims.1];
        Self {
            dims,
            row_maj: true,
            data,
        }
    }

    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(shape: T, data: U) -> Self {
        // turn a row-major vector of values into a matrix
        let shape: MatrixShape = shape.into();
        let dims = (shape.nrow, shape.ncol);
        let data: Vec<f64> = data.into_iter().collect();
        if (dims.0 * dims.1) != data.len() {
            panic!("bad shape {:?} for data length {}", dims, data.len())
        }
        Self {
            dims,
            row_maj: true,
            data,
        }
    }
}

impl PartialEq for LinearMatrix {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        if self.row_maj == other.row_maj {
            // data is arranged the same
            return self.data == other.data;
        }
        // if the data orders don't align, check the slow way
        self.flat().zip(other.flat()).all(|(x, y)| x == y)
    }
}

impl fmt::Display for LinearMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (nrow, ncol) = self.shape();
        write!(f, "rows: {} cols: {}\n", nrow, ncol)?;
        for i in 0..nrow {
            for x in self.row(i) {
                write!(f, "{:1.5} ", x)?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl<'a> Add<&'a LinearMatrix> for &'a LinearMatrix {
    type Output = LinearMatrix;
    fn add(self, rhs: Self) -> Self::Output {
        if self.dims != rhs.dims {
            panic!("incompatible shapes")
        }
        LinearMatrix::from_flat(self.dims, self.flat().zip(rhs.flat()).map(|(x, y)| x + y))
    }
}

impl<'a> Sub<&'a LinearMatrix> for &'a LinearMatrix {
    type Output = LinearMatrix;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.dims != rhs.dims {
            panic!("incompatible shapes")
        }
        LinearMatrix::from_flat(self.dims, self.flat().zip(rhs.flat()).map(|(x, y)| x - y))
    }
}

impl AddAssign<f64> for LinearMatrix {
    fn add_assign(&mut self, other: f64) {
        self.data.iter_mut().for_each(|x| *x += other);
    }
}

impl AddAssign<Self> for LinearMatrix {
    fn add_assign(&mut self, other: Self) {
        if self.dims != other.dims {
            panic!("incompatible shapes")
        }

        if self.row_maj == other.row_maj {
            self.data
                .iter_mut()
                .zip(other.data)
                .for_each(|(x, y)| x.add_assign(y));
        } else {
            self.add_ass(&other);
        }
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
            unsafe { self.data.get_unchecked(i) }
        } else {
            panic!("matrix index out of bounds")
        }
    }
}

impl IndexMut<(usize, usize)> for LinearMatrix {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        if let Some(i) = self.pos(loc) {
            unsafe { self.data.get_unchecked_mut(i) }
        } else {
            panic!("matrix index out of bounds")
        }
    }
}

impl From<Vec<f64>> for LinearMatrix {
    fn from(v: Vec<f64>) -> Self {
        // build a column vector
        Self::from_flat((v.len(), 1), v)
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
            unsafe { Some(self.data.get_unchecked(i)) }
        } else if (loc.0 < self.dim) && (loc.1 < self.dim) {
            Some(&0.0)
        } else {
            None
        }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        if let Some(i) = self.pos(loc) {
            unsafe { Some(self.data.get_unchecked_mut(i)) }
        } else {
            None
        }
    }

    fn transpose(&mut self) {
        unimplemented!() // cannot transpose this type in place
    }

    fn zeros<T: Into<MatrixShape>>(shape: T) -> Self {
        let shape: MatrixShape = shape.into();
        let dim = shape.max_dim;
        let data = vec![0.0; ((dim + 1) * (dim + 2)) / 2];
        Self { dim, data }
    }

    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(_shape: T, _data: U) -> Self {
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
            unsafe { self.data.get_unchecked(i) }
        } else if (loc.0 < self.dim) && (loc.1 < self.dim) {
            &0.0
        } else {
            panic!("matrix index out of bounds")
        }
    }
}

impl IndexMut<(usize, usize)> for LowerTriangular {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        if let Some(i) = self.pos(loc) {
            unsafe { self.data.get_unchecked_mut(i) }
        } else {
            panic!("matrix index out of bounds")
        }
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
            unsafe { Some(self.data.get_unchecked(i)) }
        } else if (loc.0 < self.dim) && (loc.1 < self.dim) {
            Some(&0.0)
        } else {
            None
        }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        if let Some(i) = self.pos(loc) {
            unsafe { Some(self.data.get_unchecked_mut(i)) }
        } else {
            None
        }
    }

    fn transpose(&mut self) {
        unimplemented!() // cannot transpose this type in place
    }

    fn zeros<T: Into<MatrixShape>>(shape: T) -> Self {
        let shape: MatrixShape = shape.into();
        let dim = shape.max_dim;
        let data = vec![0.0; ((dim + 1) * (dim + 2)) / 2];
        Self { dim, data }
    }

    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(_shape: T, _data: U) -> Self {
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
            unsafe { self.data.get_unchecked(i) }
        } else if (loc.1 < self.dim) && (loc.0 < self.dim) {
            &0.0
        } else {
            panic!("matrix index out of bounds")
        }
    }
}

impl IndexMut<(usize, usize)> for UpperTriangular {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        if let Some(i) = self.pos(loc) {
            unsafe { self.data.get_unchecked_mut(i) }
        } else {
            panic!("matrix index out of bounds")
        }
    }
}

// *****
// LowerTriangular Implementations
// *****

impl MatrixLike for Diagonal {
    fn shape(&self) -> (usize, usize) {
        (self.data.len(), self.data.len())
    }

    fn get(&self, loc: (usize, usize)) -> Option<&f64> {
        if loc.0 == loc.1 && loc.0 < self.data.len() {
            unsafe { Some(self.data.get_unchecked(loc.0)) }
        } else if loc.0 < self.data.len() && loc.1 < self.data.len() {
            Some(&0.0)
        } else {
            None
        }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        if loc.0 == loc.1 && loc.0 < self.data.len() {
            unsafe { Some(self.data.get_unchecked_mut(loc.0)) }
        } else {
            None
        }
    }

    fn transpose(&mut self) {}

    fn zeros<T: Into<MatrixShape>>(shape: T) -> Self {
        let shape = shape.into().to_rc();
        assert_eq!(shape.0, shape.1, "requested diagonal shape is not square");
        Self {
            data: vec![0.0; shape.0],
        }
    }

    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(_shape: T, _data: U) -> Self {
        // don't feel like it
        unimplemented!()
    }
}

impl PartialEq for Diagonal {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl fmt::Display for Diagonal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.disp())
    }
}

impl Index<(usize, usize)> for Diagonal {
    type Output = f64;
    fn index(&self, loc: (usize, usize)) -> &Self::Output {
        match self.get(loc) {
            Some(x) => x,
            None => panic!("index out of bounds"),
        }
    }
}

impl Index<usize> for Diagonal {
    type Output = f64;
    fn index(&self, loc: usize) -> &Self::Output {
        match self.data.get(loc) {
            Some(x) => x,
            None => panic!("index out of bounds"),
        }
    }
}

impl IndexMut<(usize, usize)> for Diagonal {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        match self.get_mut(loc) {
            Some(x) => x,
            None => panic!("index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Diagonal {
    fn index_mut(&mut self, loc: usize) -> &mut Self::Output {
        match self.data.get_mut(loc) {
            Some(x) => x,
            None => panic!("index out of bounds"),
        }
    }
}

impl From<Vec<f64>> for Diagonal {
    fn from(data: Vec<f64>) -> Self {
        Self { data }
    }
}

// *****
// ConstMatrix Implementations
// *****

impl<const SIZE: usize> MatrixLike for ConstMatrix<SIZE> {
    fn shape(&self) -> (usize, usize) {
        self.dims
    }

    fn get(&self, loc: (usize, usize)) -> Option<&f64> {
        if let Some(i) = self.pos(loc) {
            // TODO unsafe pointer math to remove second bounds check
            Some(&self.data[i])
        } else {
            None
        }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        if let Some(i) = self.pos(loc) {
            Some(&mut self.data[i])
        } else {
            None
        }
    }

    fn transpose(&mut self) {
        // nah
        unimplemented!()
    }

    fn zeros<T: Into<MatrixShape>>(shape: T) -> Self {
        let shape: MatrixShape = shape.into();
        let dims = (shape.nrow, shape.ncol);
        assert_eq!(dims.0 * dims.1, SIZE);

        Self {
            dims,
            data: [0.0; SIZE],
        }
    }

    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(shape: T, data: U) -> Self {
        // turn a row-major vector of values into a matrix
        let shape: MatrixShape = shape.into();
        let dims = (shape.nrow, shape.ncol);

        let mut arr_data = [0.0; SIZE];
        let mut last = 0;

        for (i, x) in data.into_iter().enumerate() {
            assert!(i < SIZE, "provided iterator too long");
            arr_data[i] = x;
            last = i;
        }

        assert_eq!(last, SIZE - 1, "provided iterator too short");

        Self {
            dims,
            data: arr_data,
        }
    }
}

impl<const SIZE: usize> PartialEq for ConstMatrix<SIZE> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        return self.data == other.data;
    }
}

impl<const SIZE: usize> fmt::Display for ConstMatrix<SIZE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.disp())
    }
}

impl<const SIZE: usize> Index<(usize, usize)> for ConstMatrix<SIZE> {
    type Output = f64;
    fn index(&self, loc: (usize, usize)) -> &Self::Output {
        if let Some(i) = self.pos(loc) {
            // TODO unsafe pointer math to avoid second bounds check
            &self.data[i]
        } else {
            panic!("matrix index out of bounds")
        }
    }
}

impl<const SIZE: usize> IndexMut<(usize, usize)> for ConstMatrix<SIZE> {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        if let Some(i) = self.pos(loc) {
            &mut self.data[i]
        } else {
            panic!("matrix index out of bounds")
        }
    }
}
