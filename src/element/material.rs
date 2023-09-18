use std::str::FromStr;

pub enum ProblemType {
    // strain vector e.T in comments
    PlaneStress,  // [s_xx s_yy t_xy]
    PlaneStrain,  // [e_xx e_yy g_xy]
    Axisymmetric, // [e_xx e_yy g_xy e_zz]
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub name: &'static str,
    pub youngs: f64,
    pub poisson: f64,
    pub density: f64,
}

pub const AL6061: Material = Material {
    name: "AL6061",
    youngs: 6.89e10,
    poisson: 0.33,
    density: 2700.0,
};
pub const TEST: Material = Material {
    name: "TEST",
    youngs: 1.0,
    poisson: 0.0,
    density: 1.0,
};

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
