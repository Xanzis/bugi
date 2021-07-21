pub mod iterative;

#[cfg(test)]
mod tests {
    #[test]
    fn conjugate_gradient() {
        use crate::matrix::{CompressedRow, LinearMatrix, MatrixLike, Norm};

        let k = CompressedRow::from_flat(
            4,
            vec![
                5.0, -4.0, 1.0, 0.0, -4.0, 6.0, -4.0, 1.0, 1.0, -4.0, 6.0, -4.0, 0.0, 1.0, -4.0,
                5.0,
            ],
        );
        let r = LinearMatrix::from_flat((4, 1), vec![0.0, 1.0, 0.0, 0.0]);

        let u = super::iterative::gauss_seidel(k, r, 0.0001);

        let target = LinearMatrix::from_flat((4, 1), vec![1.6, 2.6, 2.4, 1.4]);

        assert!((&target - &u).frobenius() <= 0.01);
    }
}
