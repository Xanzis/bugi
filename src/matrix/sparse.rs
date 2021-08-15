// sparse matrix storage schemes
use super::graph::Permutation;
use super::{LinearMatrix, MatrixLike, MatrixShape};

use std::collections::HashMap;
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

// lower triangular row envelope matrix storage
// stores the shortest possible row segments left of the diagonal
// each row is required to store at least one value (which may be a numerical zero)
#[derive(Debug, Clone, PartialEq)]
pub struct LowerRowEnvelope {
    n: usize,
    data: Vec<f64>,
    row_nnz: Vec<usize>,
    row_starts: Vec<usize>,
}

// row envelope matrix storage
// stores one contiguous run of values per row
// keeping this square for now, exists to store symmetric matrices
#[derive(Debug, Clone, PartialEq)]
pub struct BiEnvelope {
    n: usize,
    data: Vec<f64>,
    row_bounds: Vec<(usize, usize)>,
    row_starts: Vec<usize>,
}

// dictionary matrix storage
// stores non-zero values in a hashmap keyed by (row, col) tuples
#[derive(Debug, Clone, PartialEq)]
pub struct Dictionary {
    shape: (usize, usize),
    data: HashMap<(usize, usize), f64>,
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

    pub fn from_labeled_vals(shape: (usize, usize), mut vals: Vec<((usize, usize), f64)>) -> Self {
        // contruct from ((row, col), val) tuples
        use std::cmp::Ordering;

        vals.sort_by(|x, y| match x.0 .0.cmp(&y.0 .0) {
            Ordering::Equal => x.0 .1.cmp(&y.0 .1),
            o => o,
        });

        // vals should now be contiguous stretches of same-row values in ascending column order
        let nz_count = vals.len();

        // TODO decide if it's worth performing a bounds check on the supplied values
        let data: Vec<f64> = vals.iter().map(|(_, x)| *x).collect();
        let col_indices: Vec<usize> = vals.iter().map(|((_, c), _)| *c).collect();

        // label the row starts
        let mut row_starts = vec![0];
        let mut cur_row = 0;
        for (i, ((r, _), _)) in vals.iter().enumerate() {
            while cur_row < *r {
                row_starts.push(i);
                cur_row += 1;
            }
        }

        // fill out any trailing empty rows
        row_starts.resize(shape.0, data.len());

        Self {
            shape,
            nz_count,
            data,
            col_indices,
            row_starts,
        }
    }
}

impl LowerRowEnvelope {
    fn row_pos(&self, row: usize) -> (usize, usize, usize) {
        // return the start, final element index, and starting column number of the
        // stored portion of the requested row

        let start = self.row_starts[row];
        let diag = start + self.row_nnz[row] - 1;
        let start_col = (row + 1) - self.row_nnz[row];

        (start, diag, start_col)
    }

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

    pub fn envelope(&self) -> Vec<usize> {
        self.row_nnz.clone()
    }

    pub fn row_stored(&self, row: usize) -> (usize, &[f64]) {
        // return the stored portion of the row, as a (starting column, slice) tuple

        let (start, diag, start_col) = self.row_pos(row);
        (start_col, &self.data[start..=diag])
    }

    pub fn row_stored_mut(&mut self, row: usize) -> (usize, &mut [f64]) {
        let (start, diag, start_col) = self.row_pos(row);
        (start_col, &mut self.data[start..=diag])
    }

    pub fn row_stored_nodiag(&self, row: usize) -> (usize, &[f64]) {
        // return the sotred portion of the row without the diagonal element

        let (start, diag, start_col) = self.row_pos(row);
        (start_col, &self.data[start..diag])
    }

    pub fn solve(&self, b: &[f64], x: &mut [f64]) {
        // solves Lx = b by forward substitution

        assert_eq!(self.n, b.len(), "shapes do not agree");
        assert_eq!(self.n, x.len(), "shapes do not agree");

        for i in 0..self.n {
            let (start_col, row_stored) = self.row_stored(i);

            let dot: f64 = row_stored
                .iter()
                .zip(start_col..)
                .map(|(val, col)| val * x[col])
                .sum();

            x[i] = (b[i] - dot) / row_stored.last().unwrap();
        }
    }

    pub fn solve_submatrix(&self, b: &[f64], x: &mut [f64], range: (usize, usize)) {
        // solves L[sub_start..sub_end][sub_start..sub_end] x = b

        assert_eq!(b.len(), x.len(), "shapes do not agree");
        assert_eq!(b.len(), range.1 - range.0, "shapes do not agree");
        if range.1 > self.n {
            panic!("requested submatrix larger than base matrix");
        }

        for (sub_i, i) in (range.0..range.1).enumerate() {
            let (start_col, row_stored) = self.row_stored(i);

            let offset = if start_col < range.0 {
                range.0 - start_col
            } else {
                0
            };

            let row_stored = &row_stored[offset..];
            let start_col = start_col + offset;

            let sub_start_col = start_col - range.0;
            let dot: f64 = row_stored
                .iter()
                .zip(sub_start_col..)
                .map(|(val, col)| val * x[col])
                .sum();

            x[sub_i] = (b[sub_i] - dot) / row_stored.last().unwrap();
        }
    }

    pub fn solve_transposed(&self, y: &[f64], x: &mut [f64]) {
        // solves L'x = y by outer product

        assert_eq!(self.n, y.len(), "shapes do not agree");
        assert_eq!(self.n, x.len(), "shapes do not agree");

        x.clone_from_slice(&y);

        for i in (0..self.n).rev() {
            if x[i] == 0.0 {
                continue;
            }

            let (start_col, row_stored) = self.row_stored(i);

            let mut row_iter = row_stored.iter();
            let s = x[i] / row_iter.next_back().unwrap();
            x[i] = s;

            // row_iter no longer contains the diagonal
            for (val, col) in row_iter.zip(start_col..) {
                x[col] -= s * val;
            }
        }
    }

    pub fn non_zero_count(&self) -> usize {
        // includes zeros within the envelope
        self.data.len()
    }

    pub fn from_bienv(b: &BiEnvelope) -> Self {
        let n = b.n;

        let mut data = Vec::new();
        let mut row_starts = Vec::new();
        let mut row_nnz = Vec::new();

        for row in 0..n {
            row_starts.push(data.len());

            let (start_col, end_col) = b.row_bounds[row];

            if start_col > row || end_col <= row {
                panic!("from_bienv only implemented for envelopes overlapping diagonal");
            }

            let nnz = (row + 1) - start_col;
            row_nnz.push(nnz);

            let b_start = b.row_starts[row];

            data.extend_from_slice(&b.data[b_start..(b_start + nnz)]);
        }

        Self {
            n,
            data,
            row_starts,
            row_nnz,
        }
    }

    pub fn add_scaled_bienv(&mut self, bmat: &BiEnvelope, scale: f64) {
        // add scale * B to self
        // only add below the diagonal, panic if b rows extend to the left of self rows

        assert_eq!(self.shape(), bmat.shape());

        for row in 0..self.n {
            let (b_start, b_row) = bmat.row_stored(row);
            let (s_start, s_row) = self.row_stored_mut(row);

            if b_start < s_start {
                panic!("supplied matrix extends past envelope");
            }

            let offset = b_start - s_start;

            s_row
                .iter_mut()
                .skip(offset)
                .zip(b_row.iter())
                .for_each(|(s, b)| *s += b * scale);
        }
    }
}

impl BiEnvelope {
    fn pos(&self, loc: (usize, usize)) -> Entry<usize> {
        let (row, col) = loc;

        // TODO add methods for lookup that use fewer comparisons
        // or take advantage of structure with method for cheap slices

        if row >= self.n || col >= self.n {
            return Entry::Oob;
        }

        // column of the first element of this row's contiguous values
        let (start_col, end_col) = self.row_bounds[row];

        if col < start_col {
            return Entry::Zero;
        }

        if col >= end_col {
            return Entry::Zero;
        }

        let offset = col - start_col;
        Entry::Data(self.row_starts[row] + offset)
    }

    fn from_row_bounds(bnd: Vec<(usize, usize)>) -> Self {
        // initialize a zero matrix with the given shape
        let n = bnd.len();
        // for now, this matrix type is always square

        let mut row_starts = vec![0];

        for &(a, b) in bnd.iter() {
            let len = b - a;
            row_starts.push(row_starts.last().unwrap() + len);
        }

        let data = vec![0.0; *row_starts.last().unwrap()];

        Self {
            n,
            row_starts,
            row_bounds: bnd,
            data,
        }
    }

    pub fn row_stored(&self, row: usize) -> (usize, &[f64]) {
        // return the first column number and a slice of the stored portion of the row

        let (start_col, end_col) = self.row_bounds[row];
        let len = end_col - start_col;

        let start_idx = self.row_starts[row];

        (start_col, &self.data[start_idx..(start_idx + len)])
    }

    pub fn mul_vec(&self, x: &[f64], y: &mut [f64]) {
        // compute y = Bx fast, without allocating
        assert_eq!(x.len(), y.len());
        assert_eq!(self.n, y.len());

        for row in 0..self.n {
            let (start_col, rs) = self.row_stored(row);

            y[row] = x[start_col..]
                .iter()
                .zip(rs.iter())
                .map(|(x, b)| x * b)
                .sum();
        }
    }
}

impl Dictionary {
    pub fn envelope(&self) -> Vec<usize> {
        // find the by-row, below-diagonal envelope of the matrix

        let mut env = vec![0; self.shape().0];

        for &(r, c) in self.data.keys() {
            if c > r {
                continue;
            }

            env[r] = env[r].max((r - c) + 1);
        }

        env
    }

    pub fn row_bounds(&self) -> Vec<(usize, usize)> {
        // find the start column (inclusive) and end (exclusive) of each row

        let mut bnd = vec![(self.shape().1, 0); self.shape().0];

        for &(r, c) in self.data.keys() {
            bnd[r].0 = bnd[r].0.min(c);
            bnd[r].1 = bnd[r].1.max(c + 1);
        }

        bnd
    }

    fn in_bounds(&self, loc: (usize, usize)) -> bool {
        loc.0 < self.shape().0 && loc.1 < self.shape().1
    }

    fn get_unchecked(&self, loc: (usize, usize)) -> Option<&f64> {
        // in the current implementation this doesn't insert zeros

        self.data.get(&loc)
    }

    fn get_unchecked_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        // in the current implementation this inserts zeros

        Some(self.data.entry(loc).or_insert(0.0))
    }

    pub fn permute(&mut self, p: &Permutation) {
        let mut new_data = HashMap::new();
        let mut temp = HashMap::new();

        std::mem::swap(&mut temp, &mut self.data);

        for ((r, c), v) in temp {
            new_data.insert((p.permute(r), p.permute(c)), v);
        }

        self.data = new_data;
    }

    pub fn edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        // directed edges of the matrix's connection graph
        // includes 'connections' along the diagonal
        self.data.keys().cloned()
    }
}

impl From<Dictionary> for LowerRowEnvelope {
    fn from(dct: Dictionary) -> Self {
        // conversion ignores all values above the diagonal

        let env = dct.envelope();
        let mut res = Self::from_envelope(env);

        for (loc, v) in dct.data.into_iter() {
            if loc.1 > loc.0 {
                continue;
            }

            res[loc] = v;
        }

        res
    }
}

impl From<Dictionary> for BiEnvelope {
    fn from(dct: Dictionary) -> Self {
        let bnd = dct.row_bounds();
        let mut res = Self::from_row_bounds(bnd);

        for (loc, v) in dct.data.into_iter() {
            res[loc] = v;
        }

        res
    }
}

impl From<Dictionary> for CompressedRow {
    fn from(dct: Dictionary) -> Self {
        Self::from_labeled_vals(dct.shape(), dct.data.into_iter().collect())
    }
}

impl From<Dictionary> for LinearMatrix {
    fn from(dct: Dictionary) -> Self {
        let mut res = LinearMatrix::zeros(dct.shape());

        for r in 0..res.shape().0 {
            for c in 0..res.shape().1 {
                res[(r, c)] = dct[(r, c)];
            }
        }

        res
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

impl MatrixLike for BiEnvelope {
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

    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(_shape: T, _vals: U) -> Self {
        // not doing this for now
        unimplemented!()
    }
}

impl MatrixLike for Dictionary {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn get(&self, loc: (usize, usize)) -> Option<&f64> {
        // in the current implementation this doesn't insert zeros
        if self.in_bounds(loc) {
            self.get_unchecked(loc).or(Some(&0.0))
        } else {
            None
        }
    }

    fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
        // in the current implementation thi inserts zeros
        if self.in_bounds(loc) {
            self.get_unchecked_mut(loc)
        } else {
            None
        }
    }

    fn transpose(&mut self) {
        unimplemented!()
    }

    fn zeros<T: Into<MatrixShape>>(shape: T) -> Self {
        Self {
            shape: shape.into().to_rc(),
            data: HashMap::new(),
        }
    }

    fn from_flat<T: Into<MatrixShape>, U: IntoIterator<Item = f64>>(shape: T, vals: U) -> Self {
        let shape = shape.into();
        let mut res = Self::zeros(shape.clone());

        let mut vals = vals.into_iter();

        for r in 0..res.shape().0 {
            for c in 0..res.shape().1 {
                let v = vals
                    .next()
                    .expect("supplied iterator contains too few elements");
                if v != 0.0 {
                    res.data.insert((r, c), v);
                }
            }
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

impl fmt::Display for BiEnvelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.disp())
    }
}

impl Index<(usize, usize)> for BiEnvelope {
    type Output = f64;
    fn index(&self, loc: (usize, usize)) -> &Self::Output {
        match self.pos(loc) {
            Entry::Data(i) => unsafe { self.data.get_unchecked(i) },
            Entry::Zero => &0.0,
            Entry::Oob => panic!("matrix index out of bounds"),
        }
    }
}

impl IndexMut<(usize, usize)> for BiEnvelope {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        match self.pos(loc) {
            Entry::Data(i) => unsafe { self.data.get_unchecked_mut(i) },
            Entry::Zero => panic!("indexmut value insertion is unimplemented for BiEnvelope"),
            Entry::Oob => panic!("matrix index out of bounds"),
        }
    }
}

impl fmt::Display for Dictionary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.disp())
    }
}

impl Index<(usize, usize)> for Dictionary {
    type Output = f64;
    fn index(&self, loc: (usize, usize)) -> &Self::Output {
        if !self.in_bounds(loc) {
            panic!(
                "index out of bounds: the shape is {:?} but the index is {:?}",
                self.shape(),
                loc
            );
        }

        self.get_unchecked(loc).unwrap_or(&0.0)
    }
}

impl IndexMut<(usize, usize)> for Dictionary {
    fn index_mut(&mut self, loc: (usize, usize)) -> &mut Self::Output {
        if !self.in_bounds(loc) {
            panic!(
                "index out of bounds: the shape is {:?} but the index is {:?}",
                self.shape(),
                loc
            );
        }

        self.get_unchecked_mut(loc)
            .expect("indexmut value insertion unimplemented for dictionary")
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
