use super::Matrix;

impl Matrix {
	pub fn solve_gausselim(&mut self, mut b: Matrix) -> Result<Matrix, String> {
		// solve the system Ax=b for x. WARNING: A (self) is not preserved
		let shape = self.shape();
		if shape.0 != shape.1 { return Err(format!("square matrix required")) }
		let dim = shape.0;
		if (b.shape().1 != 1) || (b.shape().0 != dim) {
			return Err(format!("b must have shape ({}, 1) but has shape {:?}", dim, b.shape()))
		}
		let mut x = Matrix::zeros((dim, 1));

		// triangularize the matrix
		for col in 0..dim {
			// find the row with the greatest value in the current column
			// (after finished rows)
			let max = self.col(col).skip(col).enumerate()
				.max_by( |(_, &x), (_, y)| x.partial_cmp(y).unwrap() );
			if max == None { return Err(format!("solve error 1")) }
			let (max_idx, _) = max.unwrap();

			self.swap_rows(col, max_idx);
			b.swap_rows(col, max_idx);

			// scale so the new entry on the diagonal is 1
			let scaling = self.get((col, col)).unwrap();
			self.mutate_row(col, |x| *x /= scaling);
			b.mutate_row(col, |x| *x /= scaling);

			assert_eq!(self.get((col, col)), Some(1.0));

			for row in (col+1)..dim {
				let scaling = self.get((row, col)).unwrap();
				for i in col..dim {
					let to_subtract = self.get((col, i)).unwrap() * scaling;
					self.mutate((row, i), |x| *x -= to_subtract);
				}
				let to_subtract = b.get((col, 1)).unwrap() * scaling;
				b.mutate((row, 1), |x| *x -= to_subtract);
			}
		}

		//backsubstitute triangularized matrix
		for row in (0..dim).rev() {
			let mut temp = 0.0;
			for j in (row+1)..dim {
				temp += self.get((row, j)).unwrap();
			}
			x.put((row, 1), b.get((row, 1)).unwrap() - temp);
		}

		Ok(x)
	}
}