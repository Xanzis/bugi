use crate::matrix::{CompressedRow, LinearMatrix, MatrixLike, Norm};

pub fn gauss_seidel(k: CompressedRow, r: LinearMatrix, tol: f64) -> LinearMatrix {
    // iteratively solve ku = r using the gauss seidel method
    // tol is the convergence tolerance, the upper bound on the solution residual

    // relaxation factor
    const BETA: f64 = 1.3;

    assert!(k.shape().0 == k.shape().1, "non-square k");
    assert!(r.shape().1 == 1, "r must be column vector");
    assert!(r.shape().0 == k.shape().0, "r and k dimensions disagree");

    let n = k.shape().0;

    let mut u = LinearMatrix::zeros((n, 1));

    for round in 0.. {
        let mut new_u = LinearMatrix::zeros((n, 1));

        for i in 0..n {
            // keep track of the dot products while iterating through the sparse row
            let mut diag: Option<f64> = None;
            let mut lower_acc = 0.0;
            let mut upper_acc = 0.0;

            for (col, val) in k.row_nzs(i) {
                if col < i {
                    lower_acc += new_u[(col, 0)] * val;
                } else if col > i {
                    upper_acc += u[(col, 0)] * val;
                } else {
                    diag = Some(val)
                }
            }

            let diag = diag.expect("matrix must be positive definite");

            new_u[(i, 0)] = u[(i, 0)]
                + BETA * (((r[(i, 0)] - lower_acc) - diag * u[(i, 0)]) - upper_acc) / diag;
        }

        let diff_norm = (&u - &new_u).frobenius();

        u = new_u;

        if diff_norm / u.frobenius() <= tol {
            // iterations have sufficiently converged
            eprintln!("converged in {} iterations", round);
            break;
        }

        if round > 200 {
            panic!("failed to converge");
        }
    }

    u
}
