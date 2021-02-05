use crate::matrix::MatrixLike;
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

// two-point gauss-legendre sample points on the interval -1 to +1
const GAUSS_POINTS_2: [f64; 2] = [-0.57735_02691_89626,
                                  0.57735_02691_89626];
// corresponding integration weights
const GAUSS_WEIGHTS_2: [f64; 2] = [1.0, 1.0];

// four-point gauss-legendre sample points
const GAUSS_POINTS_4: [f64; 4] = [-0.86113_63115_94053,
                                  -0.33998_10435_84856,
                                  0.33998_10435_84856,
                                  0.86113_63115_94053];
// weights
const GAUSS_WEIGHTS_4: [f64; 4] = [0.34785_48451_37454,
                                   0.65214_51548_62546,
                                   0.65214_51548_62546,
                                   0.34785_48451_37454];

// references to high- and low- precision arrays (&[f64] impls IntoIter<Item=&f64>)
// TODO maybe there's a better way to make an API for this idk
const NEWTON_COATES: [&[f64]; 2] = [&NEWTON_COATES_2, &NEWTON_COATES_4];

const GAUSS_POINTS: [&[f64]; 2] = [&GAUSS_POINTS_2, &GAUSS_POINTS_4];
const GAUSS_WEIGHTS: [&[f64]; 2] = [&GAUSS_WEIGHTS_2, &GAUSS_WEIGHTS_4];

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

pub fn nd_gauss_single<T>(func: T, dim: usize, prec:usize) -> f64
where
    T: Fn(Point) -> f64,
{
    let points = GAUSS_POINTS[prec];
    let weights = GAUSS_WEIGHTS[prec];
    let n = points.len();

    let mut indices = [0, 0, 0];
    let mut sum = 0.0;

    loop {
        let mut x: Vec<f64> = Vec::new();
        let mut w = 1.0;
        for i in 0..n {
            x.push(points[i]);
            w *= weights[i];
        }
        sum += w * func(x.try_into().unwrap());

        indices[0] += 1;
        let mut i = 0;
        while indices[i] >= n {
            indices[i] = 0;
            i += 1;
            if i >= n { return sum }
            indices[i] += 1;
        }
    }
}