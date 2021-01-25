use super::{MatrixData, MatrixShape};
use super::buffer::LinearBuffer;

pub trait Inverse<T>
    where T: MatrixData
{
    fn solve_gausselim(&mut self, b: T) -> Result<T, String>;
}

impl<T> Inverse<T> for T
    where T: MatrixData
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
        let mut x = Self::zeros((dim, 1).into());
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
}
