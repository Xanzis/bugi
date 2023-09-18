use crate::matrix::LinearMatrix;
use crate::matrix::MatrixLike;
use crate::spatial::Point;

use super::material::Material;

#[derive(Debug, Clone, Copy)]
pub enum Condition {
    PlaneStrain(f64), // strain rules hold their scaling (thickness) values
    PlaneStress(f64),
    Axisymmetric,
}

impl Condition {
    pub fn thickness(&self, global_coor: Point) -> f64 {
        match self {
            Condition::PlaneStress(t) => *t,
            Condition::PlaneStrain(t) => *t,
            Condition::Axisymmetric => global_coor[0],
        }
    }

    // length of the strain vector
    pub fn vec_len(&self) -> usize {
        match self {
            Condition::PlaneStrain(_) | Condition::PlaneStress(_) => 3,
            Condition::Axisymmetric => 4,
        }
    }

    pub fn build_b(
        &self,
        h: &[f64; 3],
        h_grads_global: &LinearMatrix,
        global_coor: Point,
    ) -> LinearMatrix {
        // construct the strain interpolation matrix b
        // such that B . u = e (B times the element displacement vector is the strain vector)

        match self {
            Condition::PlaneStress(_) | Condition::PlaneStrain(_) => {
                let mut b = LinearMatrix::zeros((3, 6));
                for i in 0..3 {
                    let dhi_dx = h_grads_global[(0, i)];
                    let dhi_dy = h_grads_global[(1, i)];

                    let u_i_x = 2 * i; // local index of element node i's x DoF
                    let u_i_y = (2 * i) + 1; // local index of element node i's y DoF

                    b[(0, u_i_x)] = dhi_dx;
                    b[(1, u_i_y)] = dhi_dy;
                    b[(2, u_i_x)] = dhi_dy;
                    b[(2, u_i_y)] = dhi_dx;
                }

                b
            }
            Condition::Axisymmetric => {
                // radius, for use in hoop strain calculation
                let r = global_coor[0];

                let mut b = LinearMatrix::zeros((4, 6));
                for i in 0..3 {
                    let dhi_dx = h_grads_global[(0, i)];
                    let dhi_dy = h_grads_global[(1, i)];
                    let h_i = h[i];

                    let u_i_x = 2 * i; // local index of element node i's x DoF
                    let u_i_y = (2 * i) + 1; // local index of element node i's y DoF

                    b[(0, u_i_x)] = dhi_dx;
                    b[(1, u_i_y)] = dhi_dy;
                    b[(2, u_i_x)] = dhi_dy;
                    b[(2, u_i_y)] = dhi_dx;
                    b[(3, u_i_x)] = h_i / r;
                }

                b
            }
        }
    }

    pub fn build_c(&self, mat: Material) -> LinearMatrix {
        // return the material matrix C for use in element calculations

        match self {
            Condition::PlaneStress(_) => {
                let mut res = LinearMatrix::eye(3);
                let v = mat.poisson;
                let e = mat.youngs;
                res[(1, 0)] = v;
                res[(0, 1)] = v;
                res[(2, 2)] = (1.0 - v) / 2.0;
                res *= e / (1.0 - (v * v));
                res
            }
            Condition::PlaneStrain(_) => {
                let mut res = LinearMatrix::eye(3);
                let v = mat.poisson;
                let e = mat.youngs;
                res[(1, 0)] = v / (1.0 - v);
                res[(0, 1)] = v / (1.0 - v);
                res[(2, 2)] = (1.0 - (2.0 * v)) / (2.0 * (1.0 - v));
                res *= (e * (1.0 - v)) / ((1.0 + v) * (1.0 - (2.0 * v)));
                res
            }
            Condition::Axisymmetric => {
                let mut res = LinearMatrix::eye(4);
                let v = mat.poisson;
                let e = mat.youngs;
                let x = v / (1.0 - v);
                for (i, j) in [(0, 1), (0, 3), (3, 1)].iter().cloned() {
                    res[(i, j)] = x;
                    res[(j, i)] = x;
                }
                res[(2, 2)] = (1.0 - (2.0 * v)) / (2.0 * (1.0 - v));
                res *= (e * (1.0 - v)) / ((1.0 + v) * (1.0 - (2.0 * v)));
                res
            }
        }
    }
}

impl ToString for Condition {
    fn to_string(&self) -> String {
        match self {
            Condition::PlaneStress(t) => format!("planestress {:3.5}", t),
            Condition::PlaneStrain(t) => format!("planestrain {:3.5}", t),
            Condition::Axisymmetric => format!("axisymmetric"),
        }
    }
}
