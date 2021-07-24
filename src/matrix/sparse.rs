// sparse matrix storage schemes
use super::{MatrixLike, MatrixShape};

use std::fmt::{self, Debug};
use std::ops::{Index, IndexMut};

#[derive(Debug)]
enum Entry<T: Debug> {
    Oob,
    Zero,
    Data(T),
}

// sparse matrix struct definitions

// compressed row matrix storage
// rows are contiguous pairs of values and column indices
#[derive(Debug, Clone, PartialEq)]
pub struct CompressedRow {
    shape: (usize, usize),
    nz_count: usize, // nonzero count
    data: Vec<f64>,
    col_indices: Vec<usize>,
    row_starts: Vec<usize>,
}

// lower triangular row enelope matrix storage
// stores the shortest possible row segments left of the diagonal
// each row is required to store at least one value (which may be a numerical zero)
#[derive(Debug, Clone, PartialEq)]
pub struct LowerRowEnvelope {
    n: usize,
    data: Vec<f64>,
    row_nnz: Vec<usize>,
    row_starts: Vec<usize>,
}

// specific implementations

impl CompressedRow {
    fn pos(&self, loc: (usize, usize)) -> Entry<usize> {
        let (row, col) = loc;

        if (col >= self.shape.1) || (row >= self.shape.0) {
            return Entry::Oob;
        }

        let rstart = self.row_starts[row];
        // rend is one after last row entry
        let rend = self
            .row_starts
            .get(row + 1)
            .cloned()
            .unwrap_or(self.nz_count);

        // TODO could turn into binary search if columns are guaranteed sorted
        for i in rstart..rend {
            if self.col_indices[i] == col {
                return Entry::Data(i);
            }
        }

        return Entry::Zero;
    }

    #[allow(dead_code)]
    pub fn non_zero_count(&self) -> usize {
        // number of nonzero entries, also len of data and col_indices
        self.nz_count
    }

    pub fn row_nzs(&self, row: usize) -> CompressedRowRow {
        let rstart = self.row_starts[row];
        // rend is one after last row entry
        let rend = self
            .row_starts
            .get(row + 1)
            .cloned()
            .unwrap_or(self.nz_count);

        CompressedRowRow {
            source: &self,
            cur: rstart,
            end: rend,
        }
    }
}

impl LowerRowEnvelope {
    fn pos(&self, loc: (usize, usize)) -> Entry<usize> {
        let (row, col) = loc;

        // TODO add methods for lookup that use fewer comparisons
        // or take advantage of structure with method for cheap slices

        if row >= self.n || col >= self.n {
            return Entry::Oob;
        }

        if col > row {
            return Entry::Zero;
        }

        // column of the first element of this row's contiguous values
        let start_col = (row + 1) - self.row_nnz[row];

        if col < start_col {
            return Entry::Zero;
        }

        let offset = col - start_col;
        Entry::Data(self.row_starts[row] + offset)
    }

    pub fn from_envelope(env: Vec<usize>) -> Self {
        let n = env.len();

        if env.iter().enumerate().any(|(r, &x)| x > r + 1) {
            panic!("envelope row oversized for lower triangular matrix")
        }

        let data = vec![0.0; env.iter().cloned().sum()];

        let mut row_starts = vec![0];
        for x in env.iter().cloned() {
            if x == 0 {
                panic!("LowerRowEnvelope buffer requires nonzero envelopes for every row");
            }
            row_starts.push(row_starts.last().unwrap() + x);
        }

        row_starts.pop();

        LowerRowEnvelope {
            n,
            data,
            row_nnz: env,
            row_starts,
        }
    }

    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        // solves Lx = b by forward substitution

        if b.len() != self.n {
            panic!("shapes do not agree")
        }

        let mut x = vec![0.0; self.n];

        for i in 0..self.n {
            let row_start = self.row_starts[i];
            let diag_idx = row_start + self.row_nnz[i] - 1;
            let start_col = (i + 1) - self.row_nnz[i];

            let dot: f64 = (row_start..diag_idx)
                .zip(start_col..)
                .map(|(i, col)| self.data[i] * x[col])
                .sum();

            x[i] = (b[i] - dot) / self.data[diag_idx];
        }

        x
    }

    #[allow(dead_code)]
    pub fn non_zero_count(&self) -> usize {
        // includes zeros within the envelope
        self.data.len()
    }
}

// matrixlike implementations

impl MatrixLike for CompressedRow {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn get(&self, loc: (usize, usize)) -> Option<&f64> {
        match self.pos(loc) {
            Entry::Data(i) => unsafe { Some(self.data.get_unchecked(i)) },
            Entry::Zero => Some(&0.0),
            Entry::Oob => None,
        }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        if let Entry::Data(i) = self.pos(loc) {
            unsafe { Some(self.data.get_unchecked_mut(i)) }
        } else {
            None
        }
    }

    fn transpose(&mut self) {
        // aaaaaaa this is expensive and really shouldn't happen
        unimplemented!()
    }

    fn zeros<T: Into<MatrixShape>>(shape: T) -> Self {
        let shape: MatrixShape = shape.into();
        let shape = (shape.nrow, shape.ncol);
        Self {
            shape,
            nz_count: 0,
            data: vec![],
            col_indices: vec![],
            row_starts: vec![],
        }
    }

    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(shape: T, vals: U) -> Self {
        let shape: MatrixShape = shape.into();
        let shape = (shape.nrow, shape.ncol);

        let mut vals = vals.into_iter();

        let mut nz_count = 0;
        let mut data = vec![];
        let mut col_indices = vec![];
        let mut row_starts = vec![];

        for _row in 0..shape.0 {
            row_starts.push(data.len());
            for col in 0..shape.1 {
                let val = vals
                    .next()
                    .expect("supplied iterator does not match matrix shape");
                if val != 0.0 {
                    data.push(val);
                    col_indices.push(col);
                    nz_count += 1;
                }
            }
        }

        Self {
            shape,
            nz_count,
            data,
            col_indices,
            row_starts,
        }
    }

    // specialized mul, taking advantage of sparsity

    fn mul<T: MatrixLike, U: MatrixLike>(&self, other: &T) -> U {
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
                let row_begin = self.row_starts[r];
                let row_end = self
                    .row_starts
                    .get(r + 1)
                    .cloned()
                    .unwrap_or(self.data.len());

                let dot = (row_begin..row_end)
                    .map(|i| self.data[i] * other[(self.col_indices[i], c)])
                    .sum();

                res_vals.push(dot);
            }
        }

        U::from_flat(res_shape, res_vals)
    }
}

impl MatrixLike for LowerRowEnvelope {
    fn shape(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn get(&self, loc: (usize, usize)) -> Option<&f64> {
        match self.pos(loc) {
            Entry::Data(i) => unsafe { Some(self.data.get_unchecked(i)) },
            Entry::Zero => Some(&0.0),
            Entry::Oob => None,
        }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        if let Entry::Data(i) = self.pos(loc) {
            unsafe { Some(self.data.get_unchecked_mut(i)) }
        } else {
            None
        }
    }

    fn transpose(&mut self) {
        unimplemented!()
    }

    fn zeros<T: Into<MatrixShape>>(_shape: T) -> Self {
        // will never want this, always want to initialize with envelope
        unimplemented!()
    }

    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(shape: T, vals: U) -> Self {
        // this implementation is expensive and shouldn't be used at scale

        let shape: MatrixShape = shape.into();
        if shape.nrow != shape.ncol {
            panic!("non-square triangular matrix requested");
        }
        let n = shape.ncol;

        let mut vals = vals.into_iter();

        let mut envelope: Vec<usize> = Vec::new();
        let mut to_set: Vec<((usize, usize), f64)> = Vec::new();

        for row in 0..n {
            let mut in_env = false;
            let mut row_nnz = 0;

            for col in 0..n {
                let val = vals
                    .next()
                    .expect("supplied iterator contains too few elements");

                // woo logic
                if in_env {
                    if col > row {
                        if val != 0.0 {
                            panic!("nonzero in upper triangle")
                        }

                        in_env = false;
                    } else {
                        row_nnz += 1;
                        to_set.push(((row, col), val));
                    }
                } else {
                    if col > row {
                        if val != 0.0 {
                            panic!("nonzero in upper triangle")
                        }
                    } else {
                        if val != 0.0 {
                            row_nnz += 1;
                            to_set.push(((row, col), val));
                            in_env = true;
                        }
                    }
                }
            }

            envelope.push(row_nnz);
        }

        let mut res = LowerRowEnvelope::from_envelope(envelope);

        for (loc, val) in to_set {
            res[loc] = val;
        }

        res
    }
}

// trait implementations to satisfy matrixlike bounds

impl fmt::Display for CompressedRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.disp())
    }
}

impl Index<(usize, usize)> for CompressedRow {
    type Output = f64;
    fn index(&self, loc: (usize, usize)) -> &Self::Output {
        match self.pos(loc) {
            Entry::Data(i) => unsafe { self.data.get_unchecked(i) },
            Entry::Zero => &0.0,
            Entry::Oob => panic!("matrix index out of bounds"),
        }
    }
}

impl IndexMut<(usize, usize)> for CompressedRow {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        match self.pos(loc) {
            Entry::Data(i) => unsafe { self.data.get_unchecked_mut(i) },
            Entry::Zero => panic!("indexmut value insertion is unimplemented for CompressedRow"),
            Entry::Oob => panic!("matrix index out of bounds"),
        }
    }
}

impl fmt::Display for LowerRowEnvelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.disp())
    }
}

impl Index<(usize, usize)> for LowerRowEnvelope {
    type Output = f64;
    fn index(&self, loc: (usize, usize)) -> &Self::Output {
        match self.pos(loc) {
            Entry::Data(i) => unsafe { self.data.get_unchecked(i) },
            Entry::Zero => &0.0,
            Entry::Oob => panic!("matrix index out of bounds"),
        }
    }
}

impl IndexMut<(usize, usize)> for LowerRowEnvelope {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        match self.pos(loc) {
            Entry::Data(i) => unsafe { self.data.get_unchecked_mut(i) },
            Entry::Zero => panic!("indexmut value insertion is unimplemented for LowerRowEnvelope"),
            Entry::Oob => panic!("matrix index out of bounds"),
        }
    }
}

pub struct CompressedRowRow<'a> {
    source: &'a CompressedRow,
    cur: usize,
    end: usize,
}

impl<'a> Iterator for CompressedRowRow<'a> {
    type Item = (usize, f64);

    fn next(&mut self) -> Option<(usize, f64)> {
        if self.cur >= self.end {
            return None;
        }

        let val = self.source.data[self.cur];
        let col = self.source.col_indices[self.cur];
        self.cur += 1;
        Some((col, val))
    }
}
