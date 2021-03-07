#[derive(Debug, Clone, Copy)]
pub enum StrainRule {
	Bar,
	PlaneStrain,
	PlaneStress,
	ThreeDimensional,
}

impl StrainRule {
	// length of the strain vector
	pub fn vec_len(&self) -> usize {
		match self {
			StrainRule::Bar => 1,
			StrainRule::PlaneStrain | StrainRule::PlaneStress => 3,
			StrainRule::ThreeDimensional => 6,
		}
	}

	// given the derivative term
	// d (u, v, w)[num] / d (x, y, z)[den]
	// return the index in the strain vector to which the term should be added
	pub fn dest_idx(&self, num: usize, den: usize) -> Option<usize> {
		match self {
			StrainRule::Bar => {
				match (num, den) {
					(0, 0) => Some(0),
					_ => None,
				}
			},
			StrainRule::PlaneStrain | StrainRule::PlaneStress => {
				match (num, den) {
					(0, 0) => Some(0),
					(1, 1) => Some(1),
					(0, 1) => Some(2),
					(1, 0) => Some(2),
					_ => None,
				}
			},
			StrainRule::ThreeDimensional => {
				match (num, den) {
					(0, 0) => Some(0),
					(1, 1) => Some(1),
					(2, 2) => Some(2),
					(0, 1) => Some(3),
					(1, 0) => Some(3),
					(1, 2) => Some(4),
					(2, 1) => Some(4),
					(0, 2) => Some(5),
					(2, 0) => Some(5),
					_ => None,
				}
			}
		}
	}
}