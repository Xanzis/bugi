use super::{LowerTriangular, MatrixLike, UpperTriangular, MatrixError};

pub trait Inverse<T>
where
    T: MatrixLike,
{
    fn solve_gausselim(&mut self, b: T) -> Result<T, MatrixError>;
    fn lu_decompose(&self) -> (LowerTriangular, UpperTriangular);
    fn det_lu(&self) -> f64; // find the determinant by lu decomposition
    fn determinant(&self) -> f64;
    fn inv_lu(&self) -> T;
    fn inverse(&self) -> T;
}

impl<T> Inverse<T> for T
where
    T: MatrixLike,
{
    fn solve_gausselim(&mut self, mut b: T) -> Result<T, MatrixError> {
        // solve the system Ax=b for x. WARNING: A (self) is not preserved
        let shape = self.shape();
        if shape.0 != shape.1 {
            return Err(MatrixError::solve("square matrix required"));
        }
        let dim = shape.0;
        if (b.shape().1 != 1) || (b.shape().0 != dim) {
            return Err(MatrixError::solve(format!(
                "b must have shape ({}, 1) but has shape {:?}",
                dim,
                b.shape()
            )));
        }

        // triangularize the matrix
        for col in 0..dim {
            // find the row with the greatest value in the current column
            // (after finished rows)
            let max = self
                .col(col)
                .enumerate()
                .skip(col)
                .max_by(|(_, &x), (_, y)| x.partial_cmp(y).unwrap());
            if max == None {
                return Err(MatrixError::solve("no non-zero values remain in column"));
            }
            let (max_idx, _) = max.unwrap();

            self.swap_rows(col, max_idx);
            b.swap_rows(col, max_idx);

            // scale so the new entry on the diagonal is 1
            let scaling = self[(col, col)];
            self.mutate_row(col, |x| *x /= scaling);
            b.mutate_row(col, |x| *x /= scaling);

            for row in (col + 1)..dim {
                let scaling = self[(row, col)];
                for i in col..dim {
                    self[(row, i)] -= self[(col, i)] * scaling;
                }
                b[(row, 0)] -= b[(col, 0)] * scaling;
            }
        }

        //backsubstitute triangularized matrix
        let mut x = Self::zeros((dim, 1));
        for row in (0..dim).rev() {
            let mut temp = 0.0;
            for j in (row + 1)..dim {
                temp += self[(row, j)] * x[(j, 0)];
            }
            x[(row, 0)] = b[(row, 0)] - temp;
        }

        Ok(x)
    }

    fn lu_decompose(&self) -> (LowerTriangular, UpperTriangular) {
        let shape = self.shape();
        if shape.0 != shape.1 {
            panic!("non-square matrix");
        }
        let dim = shape.0;
        let mut lower = LowerTriangular::zeros((dim, dim));
        let mut upper = UpperTriangular::zeros((dim, dim));

        for i in 0..dim {
            for k in i..dim {
                let sum = upper.col(k).dot(lower.row(i).take(i)).end();
                upper[(i, k)] = self[(i, k)] - sum;
            }

            for k in i..dim {
                if i == k {
                    lower[(i, i)] = 1.0;
                } else {
                    let sum = lower.row(k).dot(upper.col(i)).end();
                    lower[(k, i)] = (self[(k, i)] - sum) / upper[(i, i)];
                }
            }
        }
        (lower, upper)
    }

    fn det_lu(&self) -> f64 {
        let (_, u) = self.lu_decompose();
        u.diag().product()
    }

    fn determinant(&self) -> f64 {
        // use the cheaper algorithm if possible
        if self.shape() == (2, 2) {
            self[(0, 0)] * self[(1, 1)] - self[(1, 0)] * self[(0, 1)]
        } else {
            self.det_lu()
        }
    }

    fn inv_lu(&self) -> Self {
        // find the inverse matrix via LU substitution
        let (l, u) = self.lu_decompose();
        let l_inv = l.tri_inv();
        let u_inv = u.tri_inv();

        u_inv.mul(&l_inv)
    }

    fn inverse(&self) -> Self {
        if self.shape() == (1, 1) {
            return Self::from_flat((1, 1), vec![1.0 / self[(0, 0)]]);
        }

        if self.shape() == (2, 2) {
            let det = self.determinant();

            let a = self[(0, 0)] / det;
            let b = self[(0, 1)] / det;
            let c = self[(1, 0)] / det;
            let d = self[(1, 1)] / det;

            return Self::from_flat((2, 2), vec![d, -1.0 * b, -1.0 * c, a]);
        }

        self.inv_lu()
    }
}
