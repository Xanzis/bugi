pub mod direct;
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

    #[test]
    fn envelope_cholesky() {
        use crate::matrix::{LinearMatrix, LowerRowEnvelope, MatrixLike, Norm};

        // testing cholesky decomposition for correctness
        // don't copy this usage pattern for speed-critical operations
        // dense operations and the default mul implementation are slow

        let m = LowerRowEnvelope::from_flat(
            4,
            vec![
                5.0, 0.0, 0.0, 0.0, -4.0, 6.0, 0.0, 0.0, 1.0, -4.0, 6.0, 0.0, 0.0, 1.0, -4.0, 5.0,
            ],
        );

        let l = super::direct::cholesky_envelope(&m);

        let mut l_transpose = LinearMatrix::from_flat(4, l.flat().cloned());
        l_transpose.transpose();

        let m_remade: LinearMatrix = l.mul(&l_transpose);
        let target = LinearMatrix::from_flat(
            4,
            vec![
                5.0, -4.0, 1.0, 0.0, -4.0, 6.0, -4.0, 1.0, 1.0, -4.0, 6.0, -4.0, 0.0, 1.0, -4.0,
                5.0,
            ],
        );

        assert!((&target - &m_remade).frobenius() <= 1e-8);
    }
}
