use crate::matrix::LinearMatrix;
use crate::matrix::MatrixLike;
use crate::spatial::Point;

#[derive(Debug, Clone, Copy)]
pub enum StrainRule {
    Bar,
    PlaneStrain,
    PlaneStress,
    Axisymmetric,
}

impl StrainRule {
    // length of the strain vector
    pub fn vec_len(&self) -> usize {
        match self {
            StrainRule::Bar => 1,
            StrainRule::PlaneStrain | StrainRule::PlaneStress => 3,
            StrainRule::Axisymmetric => 4,
        }
    }

    // given the derivative term
    // d (u, v, w)[num] / d (x, y, z)[den]
    // return the index in the strain vector to which the term should be added
    pub fn dest_idx(&self, num: usize, den: usize) -> Option<usize> {
        match self {
            StrainRule::Bar => match (num, den) {
                (0, 0) => Some(0),
                _ => None,
            },
            StrainRule::PlaneStrain | StrainRule::PlaneStress => match (num, den) {
                (0, 0) => Some(0),
                (1, 1) => Some(1),
                (0, 1) => Some(2),
                (1, 0) => Some(2),
                _ => None,
            },
            _ => unimplemented!(),
        }
    }

    pub fn build_b(&self, h_grads_global: &LinearMatrix, global_coor: Point) -> LinearMatrix {
        // construct the strain interpolation matrix b
        // such that B . u = e (B times the element displacement vector is the strain vector)

        match self {
            StrainRule::PlaneStress | StrainRule::PlaneStrain => {
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
            _ => unimplemented!(),
        }
    }
}
