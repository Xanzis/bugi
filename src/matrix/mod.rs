use std::fmt;
use std::ops::{Add, Mul};

// rolling my own (pretty limited) matrix math
// standard order for shape is (row, col)
// m[r][c] is stored at data[r*ncol + c]
#[derive(PartialEq, Debug)]
pub struct Matrix {
	nrow: usize,
	ncol: usize,
	data: Vec<f64>,
}

struct MatrixRow<'a> {
	source: &'a Matrix,
	row: usize,
	pos: usize,
}

struct MatrixCol<'a> {
	source: &'a Matrix,
	col: usize,
	pos: usize,
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rows: {} cols: {}", self.nrow, self.ncol)?;
        for i in 0..self.nrow {
        	for j in 0..self.ncol {
        		write!(f, "{:2.5} ", self.get((i, j)).unwrap())?;
        	}
        	write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Matrix {
	fn init(shape: (usize, usize)) -> Self {
		let mut data = Vec::new();
		data.reserve(shape.0 * shape.1);
		Matrix { nrow: shape.0, ncol: shape.1, data }
	}

	pub fn shape(&self) -> (usize, usize) {
		(self.nrow, self.ncol)
	}

	// *****
	// basic methods for value insertion / retrieval

	pub fn get_mut(&mut self, loc: (usize, usize)) -> Option<&mut f64> {
		if (loc.0 < self.nrow) && (loc.1 < self.ncol) {
			self.data.get_mut(self.ncol * loc.0 + loc.1)
		}
		else {
			None
		}
	}

	pub fn get(&self, loc: (usize, usize)) -> Option<f64> {
		if (loc.0 < self.nrow) && (loc.1 < self.ncol) {
			self.data.get(self.ncol * loc.0 + loc.1).cloned()
		}
		else {
			None
		}
	}

	pub fn put(&mut self, loc: (usize, usize), val: f64) {
		if let Some(x) = self.get_mut(loc) {
			*x = val;
		}
		else {
			panic!("bad location {:?} for matrix: {}", loc, self);
		}
	}

	// *****
	// methods returning iterators over rows/columns

	fn get_row(&self, i: usize) -> Option<MatrixRow> {
		if i < self.nrow {
			Some( MatrixRow {source: &self, row: i, pos: 0} )
		}
		else { None }
	}

	fn get_col(&self, i: usize) -> Option<MatrixCol> {
		if i < self.ncol {
			Some( MatrixCol {source: &self, col: i, pos: 0} )
		}
		else { None }
	}

	// *****
	// matrix initialization methods

	pub fn zeros(shape: (usize, usize)) -> Self {
		let mut res = Matrix::init(shape);
		for _ in 0..(res.nrow * res.ncol) {
			res.data.push(0.0);
		}
		res
	}

	pub fn eye(dim: usize) -> Self {
		let mut res = Matrix::init((dim, dim));
		// this is the naive way - this can certainly be done faster
		for r in 0..dim {
			for c in 0..dim {
				res.data.push( if r == c { 1.0 } else { 0.0 } );
			}
		}
		res
	}
}

// *****
// implementing common arithmetic operations with matrices

impl<'a> Mul for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
    	let a_shape = self.shape();
    	let b_shape = rhs.shape();

    	if a_shape.1 != b_shape.0 {
    		panic!("improper shapes for matrix multiplication: {:?} and {:?}", a_shape, b_shape)
    	}

    	let res_shape = (a_shape.0, b_shape.1);
    	let mut res = Matrix::init(res_shape);

    	for r in 0..res_shape.0 {
    		for c in 0..res_shape.1 {
    			let row = self.get_row(r).unwrap();
    			let col = rhs.get_col(c).unwrap();
    			let dot = row.zip(col).map(|x| x.0 * x.1).sum();
    			res.data.push(dot);
    		}
    	}
    	res
    }
}

impl<'a> Add for &'a Matrix {
	type Output = Matrix;

	fn add(self, rhs: Self) -> Self::Output {
		let a_shape = self.shape();
    	let b_shape = rhs.shape();

    	if (a_shape.0 != b_shape.0) || (a_shape.1 != b_shape.1) {
    		panic!("improper shapes for matrix addition: {:?} and {:?}", a_shape, b_shape)
    	}

    	let res_shape = (a_shape.0, a_shape.1);
    	let mut res = Matrix::init(res_shape);
    	res.data.extend(
    		self.data.iter()
    		.zip(rhs.data.iter())
    		.map(|x| x.0 + x.1));
    	res
	}
}

impl Iterator for MatrixCol<'_> {
    type Item = f64;
    
    fn next(&mut self) -> Option<f64> {
    	if self.pos >= self.source.nrow { return None }
        let loc = self.pos * self.source.ncol + self.col;
        self.pos += 1;
        Some(self.source.data[loc])
    }
}

impl Iterator for MatrixRow<'_> {
    type Item = f64;
    
    fn next(&mut self) -> Option<f64> {
    	if self.pos >= self.source.ncol { return None }
        let loc = self.row * self.source.ncol + self.pos;
        self.pos += 1;
        Some(self.source.data[loc])
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sub_works() {
        assert!(true);
    }
}
