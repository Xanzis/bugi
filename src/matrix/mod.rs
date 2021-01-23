use std::fmt;
use std::ops::{Add, Mul};
use crate::spatial::Point;

// rolling my own (pretty limited) matrix math
// standard order for shape is (row, col)
// m[r][c] is stored at data[r*ncol + c]
#[derive(PartialEq, Debug, Clone)]
pub struct Matrix {
	nrow: usize,
	ncol: usize,
	data: Vec<f64>,
}

pub struct MatrixRow<'a> {
	source: &'a Matrix,
	row: usize,
	pos: usize,
}

pub struct MatrixCol<'a> {
	source: &'a Matrix,
	col: usize,
	pos: usize,
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rows: {} cols: {}\n", self.nrow, self.ncol)?;
        for i in 0..self.nrow {
        	for j in 0..self.ncol {
        		write!(f, "{:1.5} ", self.get((i, j)).unwrap())?;
        	}
        	write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Matrix {
	fn init(shape: (usize, usize)) -> Self {
		let data = Vec::with_capacity(shape.0 * shape.1);
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

	pub fn get_row(&self, i: usize) -> Option<MatrixRow> {
		if i < self.nrow {
			Some( MatrixRow {source: &self, row: i, pos: 0} )
		}
		else { None }
	}

	pub fn get_col(&self, i: usize) -> Option<MatrixCol> {
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

	// *****
	// methods for alternative matrix constructions

	pub fn from_rows(rows: Vec<Vec<f64>>) -> Self {
		let nrows = rows.len();
		let ncols = rows[0].len();
		let mut res = Matrix::init((nrows, ncols));
		for r in rows.iter() {
			if r.len() != ncols { panic!("inconsistent row lengths in Matrix::from_rows") }
			res.data.extend(r);
		}
		res
	}

	pub fn from_points_row(points: Vec<Point>) -> Self {
		// construct a matrix from a series of points as row vectors
		let mut rows: Vec<Vec<f64>> = Vec::new();
		let dim = points[0].dim();

		for p in points.into_iter() {
			if p.dim() != dim { panic!("inconsistent point dimensionalities in Matrix::from_points_row"); }
			rows.push(p.into());
		}

		Matrix::from_rows(rows)
	}

	pub fn add_block_below(&mut self, other: &Matrix) {
		if self.ncol != other.ncol { panic!("inconsistent row lengths in Matrix::add_block_below"); }
		self.nrow += other.nrow;
		self.data.extend(other.data.clone());
	}

	// *****
	// matrix transformation methods

	pub fn transpose(&self) -> Self {
		let mut rows: Vec<Vec<f64>> = Vec::new();
		for i in 0..self.ncol {
			rows.push(self.get_col(i).unwrap().collect());
		}

		Matrix::from_rows(rows)
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
	use super::Matrix;
    #[test]
    fn zeros() {
    	let target = Matrix::from_rows( vec![vec![0.0, 0.0]; 3] );
        assert_eq!(target, Matrix::zeros((3, 2)));
    }
    #[test]
    fn eye() {
    	let target = Matrix::from_rows( vec![vec![1.0, 0.0], vec![0.0, 1.0]] );
    	assert_eq!(target, Matrix::eye(2));
    }
    #[test]
    fn matadd() {
    	let a = Matrix::from_rows( vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 7.0]] );
    	let b = Matrix::from_rows( vec![vec![4.0, 5.0, 6.0], vec![1.0, 2.0, 3.0]] );
    	let sum = &a + &b;
    	let target = Matrix::from_rows( vec![vec![5.0, 7.0, 9.0], vec![5.0, 7.0, 10.0]] );
    	assert_eq!(sum, target);
    }
    #[test]
    fn matmul() {
    	let a = Matrix::from_rows( vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 7.0]] );
    	let b = Matrix::from_rows( vec![vec![4.0, 2.0, 2.0, 1.0], vec![1.0, 2.0, 1.0, 1.0], vec![1.0, 2.0, 1.0, 2.0]] );
    	let target = Matrix::from_rows( vec![vec![9.0, 12.0, 7.0, 9.0], vec![28.0, 32.0, 20.0, 23.0]] );
    	let prod = &a * &b;
    	assert_eq!(target, prod);
    }
    #[test]
    fn disp() {
    	let target = "rows: 1 cols: 2\n1.00000 2.00000 \n";
    	let mat = Matrix::from_rows( vec![vec![1.0, 2.0]] );
    	println!("{}", mat);
    	assert_eq!(format!("{}", mat), target)
    }
    #[test]
    fn transpose() {
    	let a = Matrix::from_rows( vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]] );
    	let a = a.transpose();
    	let b = Matrix::from_rows( vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    	assert_eq!(a, b);
    }
}
