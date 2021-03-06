use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;

// dear reader,
// the following numbers are magic
// don't touch them

// two-interval newton-coates numbers
const NEWTON_COATES_2: [f64; 3] = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0];
// four-interval newton-coates numbers
const NEWTON_COATES_4: [f64; 5] = [
    7.0 / 90.0,
    32.0 / 90.0,
    12.0 / 90.0,
    32.0 / 90.0,
    7.0 / 90.0,
];

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

// references to high- and low- precision arrays (&[f64] impls IntoIter<Item=&f64>)
// TODO maybe there's a better way to make an API for this idk
const NEWTON_COATES: [&[f64]; 2] = [&NEWTON_COATES_2, &NEWTON_COATES_4];

// gauss-legendre integration constants, by order (0 order is nonexistent in this scheme)
const GAUSS_POINTS: [&[f64]; 5] = [&[], &GAUSS_P_1, &GAUSS_P_2, &GAUSS_P_3, &GAUSS_P_4];
const GAUSS_WEIGHTS: [&[f64]; 5] = [&[], &GAUSS_W_1, &GAUSS_W_2, &GAUSS_W_3, &GAUSS_W_4];

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

#[derive(Clone, Debug)]
pub struct NdGaussSamples<'a> {
    dim: usize,
    order: usize,
    i: usize,
    len: usize,
    points: &'a [f64],
    weights: &'a [f64],
}

impl<'a> NdGaussSamples<'a> {
    pub fn new(dim: usize, order: usize) -> Self {
        if !(0 < dim && dim <= 3) {
            panic!("invalid dimension for gauss sample iterator: {}", dim)
        }

        Self {
            dim,
            order,
            i: 0,
            len: order.pow(dim as u32),
            points: GAUSS_POINTS[order],
            weights: GAUSS_WEIGHTS[order],
        }
    }
}

impl<'a> Iterator for NdGaussSamples<'a> {
    type Item = (Point, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.len {
            return None;
        }

        let idxs = [
            self.i % self.order,
            (self.i / self.order) % self.order,
            (self.i / self.order) / self.order,
        ];

        let p_vals = [
            self.points[idxs[0]],
            self.points[idxs[1]],
            self.points[idxs[2]],
        ];
        let w_vals = [
            self.weights[idxs[0]],
            self.weights[idxs[1]],
            self.weights[idxs[2]],
        ];

        let p = Point::new(&p_vals[0..self.dim]);
        let w = w_vals[0..self.dim].iter().product();

        self.i += 1;
        Some((p, w))
    }
}

// integrate via gauss-legendre method in <dim> dimensions over bounds [-1, 1]
pub fn nd_gauss_single<T>(func: T, dim: usize, order: usize) -> f64
where
    T: Fn(Point) -> f64,
{
    let mut samples = NdGaussSamples::new(dim, order);
    let mut sum = 0.0;

    while let Some((p, w)) = samples.next() {
        sum += func(p) * w;
    }

    sum
}

// same as above, this time with a matrix as integrand
// not using generics because a dense matrix will always be fine for the application
pub fn nd_gauss_mat<T>(func: T, dim: usize, order: usize) -> LinearMatrix
where
    T: Fn(Point) -> LinearMatrix,
{
    let mut samples = NdGaussSamples::new(dim, order);

    let (first_p, first_w) = samples.next().expect("empty integration sampler");
    let mut first_term = func(first_p);
    first_term *= first_w;

    let mut sum: LinearMatrix = first_term;

    while let Some((p, w)) = samples.next() {
        let mut term = func(p);
        term *= w;
        sum.add_ass(&term);
    }

    sum
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
