use crate::matrix::{Dictionary, MatrixLike};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Constraint {
    Free,
    XFixed,
    YFixed,
    XYFixed,
    AngleFixed(f64), // unit(angle) . d = 0
}

impl Constraint {
    pub fn is_plain(&self) -> bool {
        match self {
            Constraint::XFixed => true,
            Constraint::YFixed => true,
            Constraint::XYFixed => true,
            _ => false,
        }
    }

    pub fn plain_dim_struck(&self, dim: usize) -> bool {
        // return whether a constraint has removed a given degree of freedom
        match self {
            Constraint::XFixed => dim == 0,
            Constraint::YFixed => dim == 1,
            Constraint::XYFixed => true,
            _ => false,
        }
    }
}

impl FromStr for Constraint {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // parse as a string like xy, xz, y, where presence of a dim means it is struck
        match s {
            "x" => Ok(Constraint::XFixed),
            "y" => Ok(Constraint::YFixed),
            "xy" => Ok(Constraint::XYFixed),
            other => {
                if other.starts_with("angle:") {
                    other[6..]
                        .parse::<f64>()
                        .map(|x| Constraint::AngleFixed(x))
                        .map_err(|e| ())
                } else {
                    Err(())
                }
            }
        }
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Constraint::Free => write!(f, "free"),
            Constraint::XFixed => write!(f, "x"),
            Constraint::YFixed => write!(f, "y"),
            Constraint::XYFixed => write!(f, "xy"),
            Constraint::AngleFixed(theta) => write!(f, "angle:{}", theta),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DofTransform {
    mask: Vec<bool>,
    rotations: Vec<(usize, usize, f64)>,
}

impl DofTransform {
    pub fn from_constraints(cons: &[Constraint]) -> Self {
        let mut res = Self {
            mask: vec![true; cons.len() * 2],
            rotations: Vec::new(),
        };

        for (i, con) in cons.iter().copied().enumerate() {
            match con {
                Constraint::Free => continue,
                Constraint::XFixed => {
                    res.mask[i * 2] = false;
                }
                Constraint::YFixed => {
                    res.mask[i * 2 + 1] = false;
                }
                Constraint::XYFixed => {
                    res.mask[i * 2] = false;
                    res.mask[i * 2 + 1] = false;
                }
                Constraint::AngleFixed(theta) => {
                    res.mask[i * 2] = false; // by convention the struck dof is the transformed x vector
                    res.rotations.push((i * 2, i * 2 + 1, theta))
                }
            }
        }

        res
    }

    fn mask_vec(&self, xs: &[f64]) -> Vec<f64> {
        assert_eq!(
            self.mask.len(),
            xs.len(),
            "mask length does not match provided vector"
        );
        self.mask
            .iter()
            .zip(xs)
            .filter(|(&m, x)| m)
            .map(|(m, x)| *x)
            .collect()
    }

    fn unmask_vec(&self, xs: &[f64]) -> Vec<f64> {
        let mut xs = xs.iter();
        self.mask
            .iter()
            .map(|&m| if m { *xs.next().unwrap() } else { 0.0 })
            .collect()
    }

    // names come from notation in Bathe, p139, (4.42)
    pub fn bar_matrix(&self, mat: &Dictionary) -> Dictionary {
        // K_bar = T' K T

        let mut rot = Dictionary::eye(mat.shape().0);
        for (a, b, theta) in self.rotations.iter().cloned() {
            let rotation = Dictionary::rotation(mat.shape().0, a, b, theta);
            rot = rotation.mul_dict(&rot);
        }

        let rot_prime = rot.transposed();

        let res = rot_prime.mul_dict(&mat.mul_dict(&rot));

        res.mask(&self.mask)
    }

    pub fn unbar_vec(&self, xs: &[f64]) -> Vec<f64> {
        // U = T U_bar
        let mut xs = self.unmask_vec(xs);

        for (a, b, theta) in self.rotations.iter().cloned() {
            let cos = theta.cos();
            let sin = theta.sin();
            let u = xs[a];
            let v = xs[b];
            xs[a] = 0.0 + cos * u - sin * v;
            xs[b] = 0.0 + sin * u + cos * v;
        }

        xs
    }

    pub fn bar_vec(&self, xs: &[f64]) -> Vec<f64> {
        // R_bar = T' R
        let mut xs = xs.to_vec();

        for (a, b, theta) in self.rotations.iter().cloned() {
            let cos = theta.cos();
            let sin = theta.sin();
            let u = xs[a];
            let v = xs[b];
            xs[a] = 0.0 + cos * u + sin * v;
            xs[b] = 0.0 - sin * u + cos * v;
        }

        self.mask_vec(&xs)
    }
}
