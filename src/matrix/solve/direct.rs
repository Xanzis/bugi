use std::time::Instant;

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
        let (k, r, _) = sys.into_krp();
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
        eprintln!(
            "beginning envelope cholesky solution ... ({} degrees of freedom)",
            sys.dofs()
        );

        let init_envelope = sys.k_envelope().sum::<usize>();
        eprintln!("computing reordering ...");
        let t = Instant::now();

        let dofs = sys.dofs();
        sys.reduce_envelope();

        let (k, r, perm): (LowerRowEnvelope, _, _) = sys.into_krp();

        let final_envelope = k.non_zero_count();
        let reduction = 1.0 - (final_envelope as f64 / init_envelope as f64);
        eprintln!(
            "reordering computed in {}ms (envelope reduction: {:3.3}%)",
            t.elapsed().as_millis(),
            reduction * 100.0
        );

        let perm = perm.unwrap();

        Self { dofs, k, r, perm }
    }

    fn solve(self) -> Result<Vec<f64>, ()> {
        let dofs = self.dofs;

        eprintln!("computing cholesky decomposition ...");
        let t = Instant::now();

        let chol = cholesky_envelope(&self.k);

        eprintln!(
            "decomposition computed in {}ms\nsubstituting system...",
            t.elapsed().as_millis()
        );
        let t = Instant::now();

        let mut x = vec![0.0; dofs];
        solve_ll(&chol, &self.r, &mut x);

        // invert the permutation
        self.perm.permute_slice(x.as_mut_slice());

        eprintln!("substitution complete in {}ms", t.elapsed().as_millis());

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

        l.solve_submatrix(u, &mut w, (start_col, i));

        // w is computed, fill in the new row of L
        let t = (s - w.iter().map(|x| x * x).sum::<f64>()).sqrt();

        // add the row by appending t to w
        w.push(t);
        l.row_stored_mut(i).1.copy_from_slice(&w);
    }

    l
}

pub fn solve_ll(l: &LowerRowEnvelope, b: &[f64], x: &mut [f64]) {
    // solve the system (L L') x = b
    // as first L y = b
    // then     L' x = y
    // TODO: make non-allocating in the future?

    let n = l.shape().0;
    assert_eq!(n, b.len());
    assert_eq!(n, x.len());

    let mut y = vec![0.0; n];

    l.solve(b, &mut y);
    l.solve_transposed(&y, x);
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

pub fn solve_ldl(l: &LowerRowEnvelope, d: &Diagonal, b: &[f64], x: &mut [f64]) {
    // solve the system (L D L') x = b
    // as first L z = b
    // then     D y = z  (here done in-place)
    // then     L' x = y (here still using the variable z)
    // TODO: make non-allocating in the future?

    let n = l.shape().0;
    assert_eq!(l.shape(), d.shape());
    assert_eq!(n, b.len());
    assert_eq!(n, x.len());

    let mut z = vec![0.0; n];

    l.solve(b, &mut z);
    d.solve(&mut z);
    l.solve_transposed(&z, x);
}
