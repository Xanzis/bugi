use crate::matrix::Dictionary;
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
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

    pub fn transform_vec(&self, xs: &[f64]) -> Vec<f64> {
        let mut xs = xs.to_vec();
        self.rotate_vec(&mut xs);
        self.mask_vec(&xs)
    }

    pub fn untransform_vec(&self, xs: &[f64]) -> Vec<f64> {
        let mut xs = self.unmask_vec(xs);
        self.unrotate_vec(&mut xs);
        xs
    }

    pub fn transform_matrix(&self, mut mat: Dictionary) -> Dictionary {
        self.rotate_matrix(&mut mat);
        mat.mask(&self.mask)
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

    fn rotate_matrix(&self, mat: &mut Dictionary) {
        for (a, b, theta) in self.rotations.iter().cloned() {
            mat.rotate(a, b, theta);
        }
    }

    fn rotate_vec(&self, xs: &mut [f64]) {
        for (a, b, theta) in self.rotations.iter().cloned() {
            let cos = theta.cos();
            let sin = theta.sin();
            let y = xs[a];
            let w = xs[b];
            xs[a] = cos * y - sin * w;
            xs[b] = sin * w + cos * y;
        }
    }

    fn unrotate_vec(&self, xs: &mut [f64]) {
        for (a, b, theta) in self.rotations.iter().cloned() {
            let cos = theta.cos();
            let sin = theta.sin();
            let y = xs[a];
            let w = xs[b];
            xs[a] = sin * w + cos * y;
            xs[b] = cos * y - sin * w;
        }
    }
}
