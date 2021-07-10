use crate::matrix::{LinearMatrix, MatrixLike};
use std::str::FromStr;

pub enum ProblemType {
    // strain vector e.T in comments
    Bar,              // [e_xx]
    Beam,             // [k_xx]
    PlaneStress,      // [s_xx s_yy t_xy]
    PlaneStrain,      // [e_xx e_yy g_xy]
    Axisymmetric,     // [e_xx e_yy g_xy e_zz]
    ThreeDimensional, // [e_xx e_yy e_zz g_xy g_yz g_zx]
    PlateBending,     // [k_xx k_yy k_xy]
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    name: &'static str,
    youngs: f64,
    poisson: f64,
}

pub const AL6061: Material = Material {
    name: "AL6061",
    youngs: 6.89e10,
    poisson: 0.33,
};
pub const TEST: Material = Material {
    name: "TEST",
    youngs: 1.0,
    poisson: 0.0,
};

impl Material {
    fn youngs(&self) -> f64 {
        self.youngs
    }

    fn poisson(&self) -> f64 {
        self.poisson
    }

    fn inertia_moment(&self) -> Option<f64> {
        unimplemented!()
    }

    pub fn thickness(&self) -> Option<f64> {
        // used for plate bending matrices
        unimplemented!()
    }

    pub fn get_c(&self, problem: ProblemType) -> LinearMatrix {
        // return the material matrix C for use in element calculations
        // TODO: if useful, these could be stored as symmetrical matrices
        //       (there is not yet a specialized buffer for this)
        use ProblemType::*;

        match problem {
            Bar => LinearMatrix::from_flat((1, 1), vec![self.youngs()]),
            Beam => {
                if let Some(v) = self.inertia_moment() {
                    LinearMatrix::from_flat((1, 1), vec![v * self.youngs()])
                } else {
                    panic!("inertia moment required for ProblemType::Beam but not supplied")
                }
            }
            PlaneStress => {
                let mut res = LinearMatrix::eye(3);
                let v = self.poisson();
                let e = self.youngs();
                res[(1, 0)] = v;
                res[(0, 1)] = v;
                res[(2, 2)] = (1.0 - v) / 2.0;
                res *= e / (1.0 - (v * v));
                res
            }
            PlaneStrain => {
                let mut res = LinearMatrix::eye(3);
                let v = self.poisson();
                let e = self.youngs();
                res[(1, 0)] = v / (1.0 - v);
                res[(0, 1)] = v / (1.0 - v);
                res[(2, 2)] = (1.0 - (2.0 * v)) / (2.0 * (1.0 - v));
                res *= (e * (1.0 - v)) / ((1.0 + v) * (1.0 - (2.0 * v)));
                res
            }
            Axisymmetric => {
                let mut res = LinearMatrix::eye(4);
                let v = self.poisson();
                let e = self.poisson();
                let x = v / (1.0 - v);
                for (i, j) in [(0, 1), (0, 3), (3, 1)].iter().cloned() {
                    res[(i, j)] = x;
                    res[(j, i)] = x;
                }
                res[(2, 2)] = (1.0 - (2.0 * v)) / (2.0 * (1.0 - v));
                res *= (e * (1.0 - v)) / ((1.0 + v) * (1.0 - (2.0 * v)));
                res
            }
            ThreeDimensional => {
                let mut res = LinearMatrix::eye(6);
                let v = self.poisson();
                let e = self.poisson();
                let x = v / (1.0 - v);
                for (i, j) in [(0, 1), (0, 2), (2, 1)].iter().cloned() {
                    res[(i, j)] = x;
                    res[(j, i)] = x;
                }
                for i in 3..6 {
                    res[(i, i)] = (1.0 - (2.0 * v)) / (2.0 * (1.0 - v));
                }
                res *= (e * (1.0 - v)) / ((1.0 + v) * (1.0 - (2.0 * v)));
                res
            }
            PlateBending => {
                if let Some(h) = self.thickness() {
                    let mut res = LinearMatrix::eye(3);
                    let v = self.poisson();
                    let e = self.youngs();
                    res[(1, 0)] = v;
                    res[(0, 1)] = v;
                    res[(2, 2)] = (1.0 - v) / 2.0;
                    res *= (e * h * h * h) / (12.0 * (1.0 - (v * v)));
                    res
                } else {
                    panic!("thickness required for ProblemType::PlateBending but not supplied");
                }
            }
        }
    }
}

impl FromStr for Material {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mat = match s {
            "AL6061" => AL6061,
            "TEST" => TEST,
            _ => return Err(()),
        };
        Ok(mat)
    }
}

impl ToString for Material {
    fn to_string(&self) -> String {
        self.name.to_string()
    }
}
