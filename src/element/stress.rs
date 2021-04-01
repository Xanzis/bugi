use crate::matrix::{LinearMatrix, MatrixLike};
use std::cmp;
use std::ops::{Add, Div, Mul};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StressType {
    Bar,
    Plane,
    AxiSym,
    ThreeD,
}

#[derive(Debug, Clone)]
pub struct StressState {
    pub state_type: StressType,
    data: [f64; 6],
}

impl StressState {
    fn loc(&self, a: usize, b: usize) -> usize {
        // return the index of the stress element sigma_{ab} in the canonical stress vector
        // TODO this is a slight duplicate of work in strain.rs - unify these 'problem type' enums
        match self.state_type {
            StressType::Bar => {
                if a != 0 || b != 0 {
                    panic!("invalid stress element requested ({}, {})", a, b)
                }
                0
            }
            StressType::Plane => match (a, b) {
                (0, 0) => 0,
                (1, 1) => 1,
                (0, 1) | (1, 0) => 2,
                _ => panic!("invalid stress element requested ({}, {})", a, b),
            },
            StressType::AxiSym => match (a, b) {
                (0, 0) => 0,
                (1, 1) => 1,
                (0, 1) | (1, 0) => 2,
                (2, 2) => 3,
                _ => panic!("invalid stress element requested ({}, {})", a, b),
            },
            StressType::ThreeD => match (a, b) {
                (0, 0) => 0,
                (1, 1) => 1,
                (2, 2) => 2,
                (0, 1) | (1, 0) => 3,
                (1, 2) | (2, 1) => 4,
                (0, 2) | (2, 0) => 5,
                _ => panic!("invalid stress element requested ({}, {})", a, b),
            },
        }
    }

    pub fn from_vector(vector: LinearMatrix) -> Self {
        let (r, c) = vector.shape();
        if !(r == 1 || c == 1) {
            panic!("non-flat stress vector (shape is {}, {})", r, c);
        }
        let len = cmp::max(r, c);

        let mut data = [0.0; 6];

        for (i, x) in vector.flat().enumerate() {
            if i >= 6 {
                panic!("stress vector overflow");
            }
            data[i] = *x;
        }

        // deduce the state type from the length of the vector
        // assuming the problem isn't beam or plate bending
        let state_type = match len {
            1 => StressType::Bar,
            3 => StressType::Plane,
            4 => StressType::AxiSym,
            6 => StressType::ThreeD,
            _ => panic!("invalid stress vector length"),
        };

        StressState { state_type, data }
    }

    pub fn to_cauchy(&self) -> LinearMatrix {
        let dim = match self.state_type {
            StressType::Bar => 1,
            StressType::Plane => 2,
            StressType::ThreeD => 3,
            StressType::AxiSym => unimplemented!(),
        };

        let mut res = LinearMatrix::zeros(dim);
        for r in 0..dim {
            for c in r..dim {
                res[(r, c)] = self.data[self.loc(r, c)];
            }
        }

        res
    }

    pub fn von_mises(&self) -> f64 {
        // compute the von mises yield criterion
        match self.state_type {
            StressType::Bar => self.data[0],
            StressType::Plane => {
                ((self.data[0] - self.data[1]).powi(2) + 3.0 * self.data[2].powi(2)).sqrt()
            }
            StressType::ThreeD => {
                let normals = (self.data[0] - self.data[1]).powi(2)
                    + (self.data[1] - self.data[2]).powi(2)
                    + (self.data[0] - self.data[2]).powi(2);
                let shears = self.data[3].powi(2) + self.data[4].powi(2) + self.data[5].powi(2);
                (0.5 * normals + 3.0 * shears).sqrt()
            }
            StressType::AxiSym => unimplemented!(),
        }
    }
}

impl Add for StressState {
    type Output = StressState;
    fn add(self, other: Self) -> Self::Output {
        if self.state_type != other.state_type {
            panic!("state types must be equal for addition");
        }
        Self {
            state_type: self.state_type,
            data: [
                self.data[0] + other.data[0],
                self.data[1] + other.data[1],
                self.data[2] + other.data[2],
                self.data[3] + other.data[3],
                self.data[4] + other.data[4],
                self.data[5] + other.data[5],
            ],
        }
    }
}

impl Mul<f64> for StressState {
    type Output = StressState;
    fn mul(self, other: f64) -> Self::Output {
        Self {
            state_type: self.state_type,
            data: [
                self.data[0] * other,
                self.data[1] * other,
                self.data[2] * other,
                self.data[3] * other,
                self.data[4] * other,
                self.data[5] * other,
            ],
        }
    }
}

impl Div<f64> for StressState {
    type Output = StressState;
    fn div(self, other: f64) -> Self::Output {
        Self {
            state_type: self.state_type,
            data: [
                self.data[0] / other,
                self.data[1] / other,
                self.data[2] / other,
                self.data[3] / other,
                self.data[4] / other,
                self.data[5] / other,
            ],
        }
    }
}
