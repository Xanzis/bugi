use std::ops::{AddAssign, MulAssign};

use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;

// dear reader,
// the following numbers are magic
// don't touch them

// gauss-legendre sample points and weights
// on the interval -1 to +1 for various orders
const GAUSS_P_1: [f64; 1] = [0.0];
const GAUSS_W_1: [f64; 1] = [2.0];
const GAUSS_P_2: [f64; 2] = [-0.57735_02691_89626, 0.57735_02691_89626];
const GAUSS_W_2: [f64; 2] = [1.0, 1.0];
const GAUSS_P_3: [f64; 3] = [-0.77459_66692_41483, 0.0, 0.77459_66692_41483];
const GAUSS_W_3: [f64; 3] = [
    0.55555_55555_55555,
    0.88888_88888_88888,
    0.55555_55555_55555,
];
const GAUSS_P_4: [f64; 4] = [
    -0.86113_63115_94053,
    -0.33998_10435_84856,
    0.33998_10435_84856,
    0.86113_63115_94053,
];
const GAUSS_W_4: [f64; 4] = [
    0.34785_48451_37454,
    0.65214_51548_62546,
    0.65214_51548_62546,
    0.34785_48451_37454,
];

// gauss-legendre integration constants, by order (0 order is nonexistent in this scheme)
const GAUSS_POINTS: [&[f64]; 5] = [&[], &GAUSS_P_1, &GAUSS_P_2, &GAUSS_P_3, &GAUSS_P_4];
const GAUSS_WEIGHTS: [&[f64]; 5] = [&[], &GAUSS_W_1, &GAUSS_W_2, &GAUSS_W_3, &GAUSS_W_4];

// gaussian quadrature integration points for a triangle over (0,0) (1, 0) (0, 1)
// all weights are 1.0/3.0
const GAUSS_TRI_R_1: f64 = 1.0 / 6.0;
const GAUSS_TRI_R_2: f64 = 2.0 / 3.0;
const GAUSS_TRI_R_3: f64 = GAUSS_TRI_R_1;

const GAUSS_TRI_S_1: f64 = GAUSS_TRI_R_1;
const GAUSS_TRI_S_2: f64 = GAUSS_TRI_R_1;
const GAUSS_TRI_S_3: f64 = GAUSS_TRI_R_2;

pub const GAUSS_TRI_SAMPLES: [(f64, f64); 3] = [
    (GAUSS_TRI_R_1, GAUSS_TRI_S_1),
    (GAUSS_TRI_R_2, GAUSS_TRI_S_2),
    (GAUSS_TRI_R_3, GAUSS_TRI_S_3),
];

pub fn gauss_tri<T, U>(func: T) -> U
where
    T: Fn(Point) -> U,
    U: AddAssign<U> + MulAssign<f64>,
{
    let mut res = func((GAUSS_TRI_R_1, GAUSS_TRI_S_1).into());
    res += func((GAUSS_TRI_R_2, GAUSS_TRI_S_2).into());
    res += func((GAUSS_TRI_R_3, GAUSS_TRI_S_3).into());

    res *= 1.0 / 6.0; // weight for a triangular integration region
    res
}

// compute a line integral on the segment between the given points
pub fn gauss_segment_mat<T>(func: T, a: Point, b: Point, order: usize) -> LinearMatrix
where
    T: Fn(Point) -> LinearMatrix,
{
    let dist = a.dist(b);
    let mid = a.mid(b);

    let r_func = |t| mid + (b * t * 0.5) - (a * t * 0.5); // [-1, 1] -> [a, b]

    let norm_r_prime = dist / 2.0;

    let mut samples = GAUSS_POINTS[order]
        .iter()
        .cloned()
        .zip(GAUSS_WEIGHTS[order].iter().cloned());

    let (first_p, first_w) = samples.next().expect("empty integration sampler");
    let mut first_term = func(r_func(first_p)); // parameterized input to func
    first_term *= first_w;

    let mut sum: LinearMatrix = first_term;

    while let Some((p, w)) = samples.next() {
        let mut term = func(r_func(p));
        term *= w;
        sum.add_ass(&term);
    }

    sum *= norm_r_prime;
    sum
}
