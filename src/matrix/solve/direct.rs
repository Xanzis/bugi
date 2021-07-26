use crate::matrix::{LowerRowEnvelope, MatrixLike};

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
