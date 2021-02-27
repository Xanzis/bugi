use crate::matrix::{MatrixLike, LinearMatrix};
use crate::spatial::Point;
use std::convert::TryInto;

// dear reader,
// the following numbers are magic
// don't touch them


// two-interval newton-coates numbers
const NEWTON_COATES_2: [f64; 3] = [1.0 / 6.0,
                                   4.0 / 6.0,
                                   1.0 / 6.0];
// four-interval newton-coates numbers
const NEWTON_COATES_4: [f64; 5] = [7.0 / 90.0,
                                   32.0 / 90.0,
                                   12.0 / 90.0,
                                   32.0 / 90.0,
                                   7.0 / 90.0];

// gauss-legendre sample points and weights
// on the interval -1 to +1 for various orders
const GAUSS_P_1: [f64; 1] = [0.0];
const GAUSS_W_1: [f64; 1] = [2.0];
const GAUSS_P_2: [f64; 2] = [-0.57735_02691_89626,
                             0.57735_02691_89626];
const GAUSS_W_2: [f64; 2] = [1.0, 1.0];
const GAUSS_P_3: [f64; 3] = [-0.77459_66692_41483,
                             0.0,
                             0.77459_66692_41483];
const GAUSS_W_3: [f64; 3] = [0.55555_55555_55555,
                             0.88888_88888_88888,
                             0.55555_55555_55555];
const GAUSS_P_4: [f64; 4] = [-0.86113_63115_94053,
                             -0.33998_10435_84856,
                             0.33998_10435_84856,
                             0.86113_63115_94053];
const GAUSS_W_4: [f64; 4] = [0.34785_48451_37454,
                             0.65214_51548_62546,
                             0.65214_51548_62546,
                             0.34785_48451_37454];

// references to high- and low- precision arrays (&[f64] impls IntoIter<Item=&f64>)
// TODO maybe there's a better way to make an API for this idk
const NEWTON_COATES: [&[f64]; 2] = [&NEWTON_COATES_2, &NEWTON_COATES_4];

// gauss-legendre integration constants, by order (0 order is nonexistent in this scheme)
const GAUSS_POINTS: [&[f64]; 2] = [&[], &GAUSS_P_1, &GAUSS_P_2, &GAUSS_P_3, &GAUSS_P_4];
const GAUSS_WEIGHTS: [&[f64]; 2] = [&[], &GAUSS_W_1, &GAUSS_W_2, &GAUSS_W_3, &GAUSS_W_4];

// single-value integration: just for tests
pub fn newton_single<T>(func: T, bounds: (f64, f64), prec: usize) -> f64 
where
    T: Fn(f64) -> f64,
{
    let nums = NEWTON_COATES[prec];
    let n = nums.len() - 1;

    // define the interval between sample points
    let h = (bounds.1 - bounds.0) / (n as f64);

    let mut r = bounds.0;
    let mut sum = 0.0;

    for c in nums.into_iter() {
        sum += func(r) * c;
        r += h;
    }

    (bounds.1 - bounds.0) * sum
}

// integrate via gauss-legendre method in <dim> dimensions over bounds [-1, 1]
pub fn nd_gauss_single<T>(func: T, dim: usize, order: usize) -> f64
where
    T: Fn(Point) -> f64,
{
    let points = GAUSS_POINTS[order];
    let weights = GAUSS_WEIGHTS[order];
    // sanity check for the constants
    assert_eq!(points.len(), order);

    let mut indices = [0, 0, 0];
    let mut sum = 0.0;

    loop {
        let mut x: Vec<f64> = Vec::new();
        let mut w = 1.0;
        for i in 0..dim {
            x.push(points[indices[i]]);
            w *= weights[indices[i]];
        }
        sum += w * func(x.try_into().unwrap());

        indices[0] += 1;
        let mut i = 0;
        while indices[i] >= order {
            indices[i] = 0;
            i += 1;
            if i >= dim { return sum }
            indices[i] += 1;
        }
    }
}

// same as above, this time with a matrix as integrand
// not using generics because a dense matrix will always be fine for the application
pub fn nd_gauss_mat<T, U>(func: T, dim: usize, order: usize) -> LinearMatrix
where
    T: Fn(Point) -> U,
    U: LinearMatrix,
{
    // order determines number of integration points to use
    let points = GAUSS_POINTS[order];
    let weights = GAUSS_WEIGHTS[order];
    // sanity check for the constants
    assert_eq!(points.len(), order);

    let mut indices = [0, 0, 0];
    let mut sum = None;

    loop {
        // TODO this isn't the inner loop but the heap allocation probably isn't ideal
        let mut x: Vec<f64> = Vec::new();
        let mut w = 1.0;
        // set up the point x at which to evaluate the integrand
        // x is bounded by [-1, 1] in R1, R2, or R3
        for i in 0..dim {
            x.push(points[indices[i]]);
            w *= weights[indices[i]];
        }

        // update the integrand weighted sum
        let mut term = func(x.try_into().unwrap());
        term *= w;
        sum = match sum {
            Some(s) => {
                s.add_ass(term);
                Some(s)
            },
            None => {
                Some(term)
            },
        };

        // update the gauss sample point indices
        indices[0] += 1;
        let mut i = 0;
        while indices[i] >= order {
            indices[i] = 0;
            i += 1;
            if i >= dim { return sum.unwrap() }
            indices[i] += 1;
        }
    }
}