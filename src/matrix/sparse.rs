// sparse matrix storage schemes
use super::{MatrixShape, MatrixLike};

use std::fmt::{self, Debug};
use std::ops::{Index, IndexMut};

#[derive(Debug)]
enum Entry<T: Debug> {
	Oob,
	Zero,
	Data(T),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompressedRow {
	shape: (usize, usize),
	nz_count: usize, // nonzero count
	data: Vec<f64>,
	col_indices: Vec<usize>,
	row_starts: Vec<usize>,
}

impl CompressedRow {
	fn pos(&self, loc: (usize, usize)) -> Entry<usize> {
		let (row, col) = loc;

		if (col >= self.shape.1) || (row >= self.shape.0) {
			return Entry::Oob;
		}

		let rstart = self.row_starts[row];
		// rend is one after last row entry
		let rend = self.row_starts
			.get(row + 1)
			.cloned()
			.unwrap_or(self.nz_count);

		// TODO could turn into binary search if columns are guaranteed sorted
		for i in rstart..rend {
			if self.col_indices[i] == col {
				return Entry::Data(i)
			}
		}

		return Entry::Zero
	}

	fn non_zero_count(&self) -> usize {
		// number of nonzero entries, also len of data and col_indices
		self.nz_count
	}
}

impl MatrixLike for CompressedRow {
	fn shape(&self) -> (usize, usize) {
		self.shape
	}

	fn get(&self, loc: (usize, usize)) -> Option<&f64> {
		match self.pos(loc) {
			Entry::Data(i) => unsafe { Some(self.data.get_unchecked(i)) },
			Entry::Zero => Some(&0.0),
			Entry::Oob => None
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
        		let val = vals.next().expect("supplied iterator does not match matrix shape");
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
}

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