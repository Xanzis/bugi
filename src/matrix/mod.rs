pub mod buffer;

pub mod inverse;
pub mod norm;

#[cfg(test)]
mod tests;

pub use buffer::{MatrixLike, MatrixRow, MatrixCol, LinearMatrix};
//type Matrix = LinearMatrix; // default implementation

// rolling my own (pretty limited) matrix math
// standard order for shape is (row, col)
// m[r][c] is stored at data[r*ncol + c]

// data type for matrix sizes
// will eventually add fields to help initialize sparse matrices
pub struct MatrixShape {
    ncol: usize,
    nrow: usize,
}

impl From<(usize, usize)> for MatrixShape {
    fn from(dims: (usize, usize)) -> Self {
        MatrixShape{ nrow: dims.0, ncol: dims.1 }
    }
}