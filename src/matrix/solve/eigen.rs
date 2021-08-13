use crate::matrix::graph::Permutation;
use crate::matrix::solve::{direct, System};
use crate::matrix::{BiEnvelope, LinearMatrix, LowerRowEnvelope, MatrixLike, dot};

// a range known to contain an eigenvalue
// stores the number of eigenvalues under this one (same as val index)
#[derive(Clone, Copy, Debug)]
struct OneEigenRange {
    i: usize,
    lower: f64,
    upper: f64,
}

impl OneEigenRange {
    fn new(lower: f64, upper: f64, i: usize) -> Self {
        assert!(lower < upper, "invalid range");

        Self { lower, upper, i }
    }

    fn mid_val(&self) -> f64 {
        (self.lower + self.upper) / 2.0
    }
}

#[derive(Clone, Copy, Debug)]
struct ManyEigenRange {
    low_i: usize,

    // number of the eigenvalue after the last in this range
    next_i: usize,

    lower: f64,
    upper: f64,
}

impl ManyEigenRange {
    fn new(lower: f64, upper: f64, low_i: usize, next_i: usize) -> Self {
        assert!(lower < upper, "invalid range");
        assert!(next_i > low_i, "invalid content count");
        assert!(next_i - low_i > 1, "not many");

        Self {
            lower,
            upper,
            low_i,
            next_i,
        }
    }

    fn mid_val(&self) -> f64 {
        (self.lower + self.upper) / 2.0
    }

    fn split(self, n: usize) -> (EigenRange, EigenRange) {
        // split at the midpoint, with n eigenvalues (total) below the split
        use EigenRange::{Empty, Many, One};

        let mid_val = self.mid_val();

        // number of eigenvalues in the lower and upper segment
        let l_count = n - self.low_i;
        let u_count = self.next_i - n;

        let l_range = match l_count {
            0 => Empty,
            1 => One(OneEigenRange::new(self.lower, mid_val, self.low_i)),
            x => Many(ManyEigenRange::new(self.lower, mid_val, self.low_i, n)),
        };

        let u_range = match u_count {
            0 => Empty,
            1 => One(OneEigenRange::new(mid_val, self.upper, n)),
            x => Many(ManyEigenRange::new(mid_val, self.upper, n, self.next_i)),
        };

        (l_range, u_range)
    }
}

#[derive(Clone, Copy, Debug)]
enum EigenRange {
    One(OneEigenRange),
    Many(ManyEigenRange),
    Empty,
}

impl EigenRange {
    fn new(lower: f64, upper: f64, under_lower: usize, under_upper: usize) -> Self {

        assert!(under_lower < under_upper);

        if under_upper - under_lower == 1 {
            Self::One(OneEigenRange::new(
                lower, upper, under_lower
            ))
        } else {
            Self::Many(ManyEigenRange::new(
                lower, upper, under_lower, under_upper
            ))
        }
    }
}

#[derive(Clone, Debug)]
pub struct EigenSystem {
    k_mat: LowerRowEnvelope,

    // storing the full M matrix is convenient for inverse iteration
    m_mat: BiEnvelope,

    // same for K in the first step of secant iteration
    k_full: BiEnvelope,

    // permutation to get back to original ordering
    perm: Permutation,
}

impl EigenSystem {
    pub fn new(mut sys: System) -> Self {
        let dofs = sys.dofs();
        sys.reduce_envelope();

        assert!(
            sys.k_envelope().zip(sys.m_envelope()).all(|(k, m)| k >= m),
            "m envelope larger than k"
        );

        let (k_full, m_mat, _, perm): (_, _, Option<LinearMatrix>, _) = sys.into_parts();

        let k_full: BiEnvelope = k_full.expect("k matrix not provided");
        let k_mat = LowerRowEnvelope::from_bienv(&k_full);
        let m_mat = m_mat.expect("m matrix not provided");
        let perm = perm.unwrap_or_else(|| Permutation::identity(dofs));

        Self { k_mat, m_mat, k_full, perm }
    }
}

pub struct DeterminantSearcher {
    sys: EigenSystem,
    ranges: Vec<EigenRange>,

    eigenpairs: Vec<(f64, Vec<f64>)>,
}

impl DeterminantSearcher {
    pub fn new(sys: EigenSystem) -> Self {
        Self {
            sys,
            ranges: Vec::new(),
            eigenpairs: Vec::new(),
        }
    }

    pub fn eigenvalues_under(&self, l: f64) -> usize {
        // find the number of eigenvalues under the supplied value

        // find the LDL' decomposition of (K - lM)
        let mut a = self.sys.k_mat.clone();
        a.add_scaled_bienv(&self.sys.m_mat, -1.0 * l);
        let (_, d) = direct::cholesky_envelope_no_root(&a);

        // TODO add handling for too-singular K-lM (solution: just test a different value)
        d.num_neg()
    }

    fn process_one(&mut self, mut r: OneEigenRange) {
        // refine an eigenrange, then send it to inverse iteration and store the result
        // TODO improve this procedure - might need to improve guess first

        // accept the middle of the resulting range for inverse iteration
        let eigenpair = inverse_iterate(&self.sys.k_mat, &self.sys.m_mat, r.mid_val(), 1e-6);

        self.eigenpairs.push(eigenpair);

        // process_eigenrange's order should ensure eigenpairs ends up sorted by value
    }

    fn process_eigenrange(&mut self, r: EigenRange) {
        match r {
            EigenRange::One(r) => {
                self.process_one(r);
            }
            EigenRange::Many(r) => {
                let under = self.eigenvalues_under(r.mid_val());
                let (rl, ru) = r.split(under);
                // recursively following (with lower values evaulated first)
                self.process_eigenrange(rl);
                self.process_eigenrange(ru);
            }
            EigenRange::Empty => (),
        }
    }

    fn char_ratio(&self, mu_old: f64, mu_new: f64) -> f64 {
        // evaluate the characteristic polynomial ratio of the eigenproblem at mu
        // find p(mu_new) / (p(mu_new) - p(mu_old))
        // scale by p(mu_new) during calculation to keep numbers manageable

        let mu_diff = mu_new - mu_old;

        let mut a = self.sys.k_mat.clone();
        a.add_scaled_bienv(&self.sys.m_mat, -1.0 * mu_new);

        let (_, d) = direct::cholesky_envelope_no_root(&a);
        let mean_diag = d.mean();
        let p_new = d.product_scaled(1.0 / mean_diag);

        // compute the characteristic polynomial for the old shift
        // K - mu_old * M = K - mu_new * M + (mu_new - mu_old) * M
        a.add_scaled_bienv(&self.sys.m_mat, mu_diff);


        // TODO avoid recomputing p_old (can reuse previous value)
        let (_, d) = direct::cholesky_envelope_no_root(&a);
        let p_old = d.product_scaled(1.0 / mean_diag);

        p_new / (p_new - p_old)
    }

    fn next_val(&self, mu_old: f64, mu_new: f64, eta: f64) -> f64 {
        // one secant step, finding the next eigenvalue approximation to attempt
        // uses the past two approximations
        mu_new - (eta * self.char_ratio(mu_old, mu_new) * (mu_new - mu_old))
    }

    pub fn secant_iterate_one(&mut self) {
        // discover the next undiscovered eigenvalue(s) by secant iteration

        // find first two lower bounds on the first eigenvalue
        let mut mu_old = 0.0;

        // inverse iterate three times at zero shift
        let x = inverse_iterate_times(&self.sys.k_mat, &self.sys.m_mat, 3);

        let mut mu_new = 0.99 * rayleigh_quotient(x.as_slice(), &self.sys.k_full, &self.sys.m_mat);

        loop {
            let p = self.eigenvalues_under(mu_new);
            if p == 0 {
                break
            } else {
                mu_new /= p as f64;
            }
        }

        // begin the iteration
        let mut eta = 2.0;

        let eigens_under = self.eigenpairs.len();

        let range = loop {
            let temp = self.next_val(mu_old, mu_new, eta);
            mu_old = mu_new;
            mu_new = temp;

            let new_under = self.eigenvalues_under(mu_new);
            if new_under != eigens_under {
                break EigenRange::new(mu_old, mu_new, eigens_under, new_under)
            }

            // if the relative change is small, double eta
            if ((mu_old - mu_new) / mu_new).abs() < 0.01 {
                eta *= 2.0;
            }
        };

        self.process_eigenrange(range);
    }

    pub fn eigenpairs(&self) -> Vec<(f64, Vec<f64>)> {
        self.eigenpairs.iter()
            .cloned()
            .map(|(val, mut v)| {
                self.sys.perm.permute_slice(v.as_mut_slice());
                (val, v)
            })
            .collect()
    }
}

fn inverse_iterate(k: &LowerRowEnvelope, m: &BiEnvelope, shift: f64, tol: f64) -> (f64, Vec<f64>) {
    // perform inverse iteration on the shifted eigenproblem
    // (K - shift * M) * v = l * M * v

    assert_eq!(k.shape(), m.shape());
    let n = k.shape().0;

    // for now, clone the supplied K - maybe investigate nonallocating alternatives in the future
    let mut k = k.clone();
    k.add_scaled_bienv(m, -1.0 * shift);
    // for all intents and purposes (K - shift * M) is now the K to be used
    // the following algorithm iterates on the system K * v = l * M * v

    // decompose the shifted matrix (this only needs to happen once)
    let (k_l, k_d) = direct::cholesky_envelope_no_root(&k);

    let mut old_rho: Option<f64> = None;

    // begin iteration with x as all-ones
    let mut x = vec![1.0; n];
    let mut x_bar = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut y_bar = vec![0.0; n];

    // y(0) = M x(0)
    m.mul_vec(x.as_slice(), y.as_mut_slice());

    loop {
        // K x_bar(k+1) = y(k)
        direct::solve_ldl(&k_l, &k_d, y.as_slice(), x_bar.as_mut_slice());
        // y_bar(k+1) = M x_bar(k+1)
        m.mul_vec(x_bar.as_slice(), y_bar.as_mut_slice());

        // Rayleigh quotient approximation:
        // rho = x_bar(k+1) . y(k) / x_bar(k+1) . y_bar(k+1)
        let rho = dot(&x_bar, &y) / dot(&x_bar, &y_bar);

        let mag = dot(&x_bar, &y_bar).sqrt();

        if let Some(r) = old_rho {
            if ((rho - r) / rho).abs() <= tol {
                // ready to return, construct the final approximation
                for i in 0..n {
                    x[i] = x_bar[i] / mag;
                }

                // add the shift back on
                break (rho + shift, x);
            }
        }

        old_rho = Some(rho);

        // y(k+1) = y_bar(k+1) / sqrt(x_bar(k+1) . y_bar(k+1))
        for i in 0..n {
            y[i] = y_bar[i] / mag;
        }
    }
}

fn inverse_iterate_times(k: &LowerRowEnvelope, m: &BiEnvelope, times: usize) -> Vec<f64> {
    // perform a set number of inverse iterations on an unshifted problem
    // used at the beginning of secant iteration to obtain a starting shift

    assert_eq!(k.shape(), m.shape());
    let n = k.shape().0;

    let (k_l, k_d) = direct::cholesky_envelope_no_root(k);

    // begin iteration with x as all-ones
    let mut x = vec![1.0; n];
    let mut x_bar = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut y_bar = vec![0.0; n];

    let mut mag = 0.0;

    // y(0) = M x(0)
    m.mul_vec(x.as_slice(), y.as_mut_slice());

    for _ in 0..n {
        // K x_bar(k+1) = y(k)
        direct::solve_ldl(&k_l, &k_d, y.as_slice(), x_bar.as_mut_slice());
        // y_bar(k+1) = M x_bar(k+1)
        m.mul_vec(x_bar.as_slice(), y_bar.as_mut_slice());

        mag = dot(&x_bar, &y_bar).sqrt();

        // y(k+1) = y_bar(k+1) / sqrt(x_bar(k+1) . y_bar(k+1))
        for i in 0..n {
            y[i] = y_bar[i] / mag;
        }
    }

    for i in 0..n {
        x[i] = x_bar[i] / mag;
    }

    x
}

fn rayleigh_quotient(x: &[f64], k: &BiEnvelope, m: &BiEnvelope) -> f64 {
    // compute the rayleigh quotient x' K x / x' M x
    let n = k.shape().0;

    assert_eq!(k.shape(), m.shape());
    assert_eq!(n, x.len());

    let mut temp = vec![0.0; n];

    k.mul_vec(x, temp.as_mut_slice());
    let num = dot(x, &temp);

    m.mul_vec(x, temp.as_mut_slice());
    let den = dot(x, &temp);

    num / den
}