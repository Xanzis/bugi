use crate::matrix::{Dictionary, MatrixLike};

pub mod direct;
pub mod iterative;

pub trait Solver {
    fn new(sys: System) -> Self;

    // TODO improve matrix::Error API and add to this result
    fn solve(self) -> Result<Vec<f64>, ()>;
}

#[derive(Debug, Clone)]
pub struct System {
    mat: Dictionary,
    rhs: Vec<f64>,
}

impl System {
    pub fn new(dofs: usize) -> Self {
        Self {
            mat: Dictionary::zeros(dofs),
            rhs: vec![0.0; dofs],
        }
    }

    pub fn add_coefficient(&mut self, loc: (usize, usize), val: f64) {
        self.mat[loc] += val;
    }

    pub fn add_rhs_val(&mut self, loc: usize, val: f64) {
        self.rhs[loc] += val;
    }

    pub fn from_kr<T, U>(k: &T, r: &U) -> Self
    where
        T: MatrixLike,
        U: MatrixLike,
    {
        let dofs = k.shape().0;

        if dofs != k.shape().1 || dofs != r.shape().0 || r.shape().1 != 1 {
            panic!("bad k, r shapes: {:?}, {:?}", k, r);
        }

        let mut res = Self::new(dofs);

        for row in 0..dofs {
            for col in 0..dofs {
                res.mat[(row, col)] = k[(row, col)];
            }
            res.rhs[row] = r[(row, 0)];
        }

        res
    }

    fn into_parts<T, U>(self) -> (T, U)
    where
        T: From<Dictionary>,
        U: From<Vec<f64>>,
    {
        (self.mat.into(), self.rhs.into())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn gauss_seidel() {
        use crate::matrix::{CompressedRow, LinearMatrix, MatrixLike, Norm};

        let k = CompressedRow::from_flat(
            4,
            vec![
                5.0, -4.0, 1.0, 0.0, -4.0, 6.0, -4.0, 1.0, 1.0, -4.0, 6.0, -4.0, 0.0, 1.0, -4.0,
                5.0,
            ],
        );
        let r = LinearMatrix::from_flat((4, 1), vec![0.0, 1.0, 0.0, 0.0]);

        let u = super::iterative::gauss_seidel(k, r, 0.0001, 200);

        let target = LinearMatrix::from_flat((4, 1), vec![1.6, 2.6, 2.4, 1.4]);

        assert!((&target - &u).frobenius() <= 0.01);
    }

    #[test]
    fn solver_impls() {
        use super::{
            direct::CholeskyEnvelopeSolver, direct::DenseGaussSolver, iterative::GaussSeidelSolver,
            Solver, System,
        };
        use crate::matrix::{LinearMatrix, MatrixLike};

        let k = LinearMatrix::from_flat(
            4,
            vec![
                5.0, -4.0, 1.0, 0.0, -4.0, 6.0, -4.0, 1.0, 1.0, -4.0, 6.0, -4.0, 0.0, 1.0, -4.0,
                5.0,
            ],
        );
        let r = LinearMatrix::from_flat((4, 1), vec![0.0, 1.0, 0.0, 0.0]);

        let sys = System::from_kr(&k, &r);

        let target = vec![1.6, 2.6, 2.4, 1.4];

        let dg = DenseGaussSolver::new(sys.clone());
        let dg_res = dg.solve().unwrap();
        assert!(
            dg_res
                .iter()
                .zip(target.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                <= 1.0e-8
        );

        let gs = GaussSeidelSolver::new(sys.clone());
        let gs_res = gs.solve().unwrap();
        assert!(
            gs_res
                .iter()
                .zip(target.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                <= 1.0e-2
        );

        let ch = CholeskyEnvelopeSolver::new(sys.clone());
        let ch_res = ch.solve().unwrap();
        assert!(
            ch_res
                .iter()
                .zip(target.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                <= 1.0e-2
        )
    }

    #[test]
    fn envelope_cholesky() {
        use crate::matrix::{Diagonal, LinearMatrix, LowerRowEnvelope, MatrixLike, Norm};

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

        let (l, d) = super::direct::cholesky_envelope_no_root(&m);

        let mut l_transpose = LinearMatrix::from_flat(4, l.flat().cloned());
        l_transpose.transpose();

        let temp: LinearMatrix = d.mul(&l_transpose);
        let m_remade: LinearMatrix = l.mul(&temp);

        assert!((&target - &m_remade).frobenius() <= 1e-8);

        // example LDL' decomposition on non positive-definite problem

        let m = LowerRowEnvelope::from_flat(
            4,
            vec![
                2.0, 0.0, 0.0, 0.0, 6.0, 17.0, 0.0, 0.0, 0.0, 2.0, -1.0, 0.0, 6.0, 17.0, -4.0, 33.0,
            ],
        );

        let (l, d) = super::direct::cholesky_envelope_no_root(&m);

        let mut l_transpose = LinearMatrix::from_flat(4, l.flat().cloned());
        l_transpose.transpose();

        let temp: LinearMatrix = d.mul(&l_transpose);
        let m_remade: LinearMatrix = l.mul(&temp);

        let target = LinearMatrix::from_flat(
            4,
            vec![
                2.0, 6.0, 0.0, 6.0, 6.0, 17.0, 2.0, 17.0, 0.0, 2.0, -1.0, -4.0, 6.0, 17.0, -4.0,
                33.0,
            ],
        );

        assert!((&target - &m_remade).frobenius() <= 1e-8);

        let target_diagonal: Diagonal = vec![2.0, -1.0, 3.0, 4.0].into();
        let target_diagonal = LinearMatrix::from_flat(4, target_diagonal.flat().cloned());
        let d = LinearMatrix::from_flat(4, d.flat().cloned());

        assert!((&target_diagonal - &d).frobenius() <= 1e-8);
    }
}
