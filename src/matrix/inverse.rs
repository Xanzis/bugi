use super::MatrixLike;

pub trait Inverse<T>
    where T: MatrixLike
{
    fn solve_gausselim(&mut self, b: T) -> Result<T, String>;
    fn lu_decompose(&self) -> (T, T);
    fn det_lu(&self) -> f64; // find the determinant by lu decomposition
    fn determinant(&self) -> f64;
    fn inverse(&self) -> T;
}

impl<T> Inverse<T> for T
    where T: MatrixLike
{
    fn solve_gausselim(&mut self, mut b: T) -> Result<T, String> {
        // solve the system Ax=b for x. WARNING: A (self) is not preserved
        let shape = self.shape();
        if shape.0 != shape.1 {
            return Err(format!("square matrix required"));
        }
        let dim = shape.0;
        if (b.shape().1 != 1) || (b.shape().0 != dim) {
            return Err(format!(
                "b must have shape ({}, 1) but has shape {:?}",
                dim,
                b.shape()
            ));
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
                return Err(format!("solve error 1"));
            }
            let (max_idx, _) = max.unwrap();

            self.swap_rows(col, max_idx);
            b.swap_rows(col, max_idx);

            // scale so the new entry on the diagonal is 1
            let scaling = self.get((col, col)).unwrap().clone();
            self.mutate_row(col, |x| *x /= scaling);
            b.mutate_row(col, |x| *x /= scaling);

            for row in (col + 1)..dim {
                let scaling = self.get((row, col)).unwrap().clone();
                for i in col..dim {
                    let to_subtract = self.get((col, i)).unwrap() * scaling;
                    self.mutate((row, i), |x| *x -= to_subtract);
                }
                let to_subtract = b.get((col, 0)).unwrap() * scaling;
                b.mutate((row, 0), |x| *x -= to_subtract);
            }
        }

        //backsubstitute triangularized matrix
        let mut x = Self::zeros((dim, 1));
        for row in (0..dim).rev() {
            let mut temp = 0.0;
            for j in (row + 1)..dim {
                temp += self.get((row, j)).unwrap() * x.get((j, 0)).unwrap();
            }
            println!("{:?}", temp);
            x.put((row, 0), b.get((row, 0)).unwrap() - temp);
        }

        Ok(x)
    }

    fn lu_decompose(&self) -> (Self, Self) {
        let shape = self.shape();
        if shape.0 != shape.1 {
            panic!("non-square matrix");
        }
        let dim = shape.0;
        let mut lower = Self::zeros((dim, dim));
        let mut upper = Self::zeros((dim, dim));

        for i in 0..dim {
            for k in i..dim {
                let mut sum = 0.0;
                for j in 0..i {
                    sum += lower.get((i, j)).unwrap() * upper.get((j, k)).unwrap();
                }
                upper.put((i, k), self.get((i, k)).unwrap() - sum);
            }

            for k in i..dim {
                if i == k {
                    lower.put((i, i), 1.0);
                }
                else {
                    let mut sum = 0.0;
                    for j in 0..i {
                        sum += lower.get((k, j)).unwrap() * upper.get((j, i)).unwrap();
                    }
                    lower.put((k, i), (self.get((k, i)).unwrap() - sum) / upper.get((i, i)).unwrap());
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
            self.get((0, 0)).unwrap() * self.get((1, 1)).unwrap() -
            self.get((1, 0)).unwrap() * self.get((0, 1)).unwrap()
        }
        else {
            self.det_lu()
        }
    }

    fn inverse(&self) -> Self {
        if self.shape() == (2, 2) {
            let det = self.determinant();

            let a = self.get((0, 0)).unwrap() / det;
            let b = self.get((0, 1)).unwrap() / det;
            let c = self.get((1, 0)).unwrap() / det;
            let d = self.get((1, 1)).unwrap() / det;

            return Self::from_flat((2, 2), vec![d, -1.0 * b, -1.0 * c, a]);
        }

        unimplemented!()
    }
}