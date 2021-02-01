use crate::matrix::{LinearMatrix, MatrixLike};

enum ProblemType {
    // strain vector e.T in comments
    Bar,              // [e_xx]
    Beam,             // [k_xx]
    PlaneStress,      // [e_xx e_yy g_xy]
    PlaneStrain,      // [e_xx e_yy g_xy]
    Axisymmetric,     // [e_xx e_yy g_xy e_zz]
    ThreeDimensional, // [e_xx e_yy e_zz g_xy g_yz g_zx]
    PlateBending,     // [k_xx k_yy k_xy]
}

pub trait Material {
    fn youngs(&self) -> f64;
    fn poisson(&self) -> f64;

    fn inertia_moment(&self) -> Option<f64> {
        None
    }

    fn thickness(&self) -> Option<f64> {
        // used for plate bending matrices
        None
    }

    fn get_c(&self, problem: ProblemType) -> LinearMatrix {
        // return the material matrix C for use in element calculations
        // TODO: if useful, these could be stored as symmetrical matrices
        //       (there is not yet a specialized buffer for this)
        use ProblemType::*;

        match problem {
            Bar => {
                LinearMatrix::from_flat((1, 1), vec![self.youngs()])
            },
            Beam => {
                if let Some(v) = self.inertia_moment() {
                    LinearMatrix::from_flat((1, 1), vec![v*self.youngs()])
                }
                else {
                    panic!("inertia moment required for ProblemType::Beam but not supplied")
                }
            },
            PlaneStress => {
                let mut res = LinearMatrix::eye(3);
                let v = self.poisson();
                let e = self.youngs();
                res.put((1, 0), v);
                res.put((0, 1), v);
                res.put((2, 2), (1.0 - v) / 2.0);
                res *= e / (1.0 - (v * v));
                res
            }
            PlaneStrain => {
                let mut res = LinearMatrix::eye(3);
                let v = self.poisson();
                let e = self.youngs();
                res.put((1, 0), v / (1.0 - v));
                res.put((0, 1), v / (1.0 - v));
                res.put((2, 2), (1.0 - (2.0 * v)) / (2.0 * (1.0 - v)));
                res *= (e * (1.0 - v)) / ((1.0 + v) * (1.0 - (2.0 * v)));
                res
            }
            Axisymmetric => {
                let mut res = LinearMatrix::eye(4);
                let v = self.poisson();
                let e = self.poisson();
                let x = v / (1.0 - v);
                for (i, j) in [(0, 1), (0, 3), (3, 1)].iter() {
                    res.put((*i, *j), x);
                    res.put((*j, *i), x);
                }
                res.put((2, 2), (1.0 - (2.0 * v)) / (2.0 * (1.0 - v)));
                res *= (e * (1.0 - v)) / ((1.0 + v) * (1.0 - (2.0 * v)));
                res
            }
            ThreeDimensional => {
                let mut res = LinearMatrix::eye(6);
                let v = self.poisson();
                let e = self.poisson();
                let x = v / (1.0 - v);
                for (i, j) in [(0, 1), (0, 2), (2, 1)].iter() {
                    res.put((*i, *j), x);
                    res.put((*j, *i), x);
                }
                for i in 3..6 {
                    res.put((i, i), (1.0 - (2.0 * v)) / (2.0 * (1.0 - v)));
                }
                res *= (e * (1.0 - v)) / ((1.0 + v) * (1.0 - (2.0 * v)));
                res
            }
            PlateBending => {
                if let Some(h) = self.thickness() {
                    let mut res = LinearMatrix::eye(3);
                    let v = self.poisson();
                    let e = self.youngs();
                    res.put((1, 0), v);
                    res.put((0, 1), v);
                    res.put((2, 2), (1.0 - v) / 2.0);
                    res *= (e * h * h * h) / (12.0 * (1.0 - (v * v)));
                    res
                }
                else {
                    panic!("thickness required for ProblemType::PlateBending but not supplied");
                }
            }
        }
    }
}

pub struct Aluminum6061;

impl Material for Aluminum6061 {
    fn youngs(&self) -> f64 {
        // SI units: Pa
        6.89e10
    }
    fn poisson(&self) -> f64 {
        // unitless
        0.33
    }
}