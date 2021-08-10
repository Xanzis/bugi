use crate::matrix::graph::Permutation;
use crate::matrix::{Diagonal, Inverse, LinearMatrix, LowerRowEnvelope, MatrixLike};

use super::{Solver, System};

#[derive(Clone, Debug)]
pub struct DenseGaussSolver {
    k: LinearMatrix,
    r: LinearMatrix,
}

impl Solver for DenseGaussSolver {
    fn new(sys: System) -> Self {
        let (k, r, _) = sys.into_parts();
        Self { k, r }
    }

    fn solve(mut self) -> Result<Vec<f64>, ()> {
        eprintln!("beginning dense gauss solution ...");
        let res = self.k.solve_gausselim(self.r).map_err(|_| ())?;
        eprintln!("dense gauss solution complete");

        Ok(res.flat().cloned().collect())
    }
}

#[derive(Clone, Debug)]
pub struct CholeskyEnvelopeSolver {
    dofs: usize,
    k: LowerRowEnvelope,
    r: Vec<f64>,
    perm: Permutation,
}

impl Solver for CholeskyEnvelopeSolver {
    fn new(mut sys: System) -> Self {
        eprintln!("beginning envelope cholesky solution ...");

        eprintln!(
            "computing reordering (initial total envelope: {}) ...",
            sys.envelope_sum()
        );

        let dofs = sys.dofs();
        sys.reduce_envelope();

        let (k, r, perm): (LowerRowEnvelope, _, _) = sys.into_parts();

        eprintln!(
            "reordering computed (total envelope: {})",
            k.non_zero_count()
        );

        let perm = perm.unwrap();

        Self { dofs, k, r, perm }
    }

    fn solve(self) -> Result<Vec<f64>, ()> {
        let dofs = self.dofs;

        eprintln!("computing cholesky decomposition ...");

        let chol = cholesky_envelope(&self.k);

        eprintln!("decomposition computed.\nsolving system...");

        // solve the system L L' x = b as L y = b, L' x = y
        let mut y = vec![0.0; dofs];
        chol.solve(self.r.as_slice(), y.as_mut_slice());

        let mut x = vec![0.0; dofs];
        chol.solve_transposed(y.as_slice(), x.as_mut_slice());

        // invert the permutation
        self.perm.permute_slice(x.as_mut_slice());

        eprintln!("envelope cholesky solution complete");

        Ok(x)
    }
}

pub fn cholesky_envelope(a: &LowerRowEnvelope) -> LowerRowEnvelope {
    // find the cholesky factor L such that A = LL'

    // A is assumed to be a symmetric positive definite matrix
    // with only the bottom triangular portion given

    // factors A using the relation
    //     [M  u]      [P  0]
    // A = [u' s], L = [w' t]
    // where A = LL', M = PP', Pw = u, and t = (s - w'w)^0.5

    let envelope = a.envelope();

    let mut l = LowerRowEnvelope::from_envelope(envelope);

    l[(0, 0)] = a[(0, 0)].sqrt();

    let mut w = Vec::new();

    for i in 1..a.shape().0 {
        let (start_col, u) = a.row_stored(i);

        // by the contruction of LowerRowEnvelope u.len() != 0
        let nondiag_len = u.len() - 1;

        if nondiag_len == 0 {
            let t = u[0].sqrt();
            l[(i, i)] = t;
            continue;
        }

        let s = u[nondiag_len];
        let u = &u[..nondiag_len];

        w.clear();
        w.resize(nondiag_len, 0.0);

        l.solve_submatrix(u, w.as_mut_slice(), (start_col, i));

        // w is computed, fill in the new row of L
        let t = (s - w.iter().map(|x| x * x).sum::<f64>()).sqrt();

        // add the row by appending t to w
        w.push(t);
        l.row_stored_mut(i).1.clone_from_slice(w.as_slice());
    }

    l
}

pub fn cholesky_envelope_no_root(a: &LowerRowEnvelope) -> (LowerRowEnvelope, Diagonal) {
    // find the LDL' factorization of symmetric matrix A, where D may have negative elements

    let n = a.shape().0;
    let mut l = LowerRowEnvelope::from_envelope(a.envelope());
    let mut d = Diagonal::zeros(n);

    for i in 0..n {
        let (start_col, u) = l.row_stored(i);
        let dot = (start_col..i)
            .zip(u.iter())
            .map(|(j, x)| x * x * d[j])
            .sum::<f64>();
        d[(i, i)] = a[(i, i)] - dot;

        for j in (i + 1)..n {
            if l.row_stored(j).0 > i {
                continue;
            };

            let (start_i, u) = l.row_stored(i);
            let (start_j, v) = l.row_stored(j);

            // properly align the i and j rows
            let start_col = std::cmp::max(start_i, start_j);
            let u = u[(start_col - start_i)..].iter();
            let v = v[(start_col - start_j)..].iter();

            let dot = (start_col..i)
                .zip(u.zip(v))
                .map(|(k, (x, y))| x * y * d[k])
                .sum::<f64>();

            l[(j, i)] = (a[(j, i)] - dot) / d[i];
        }

        l[(i, i)] = 1.0;
    }

    (l, d)
}
