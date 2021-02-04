pub trait StrainRule {
	// length of the strain vector
	fn vec_len(&self) -> usize;

	// given the derivative term
	// d (u, v, w)[num] / d (x, y, z)[den]
	// return the index in the strain vector to which the term should be added
	fn dest_idx(&self, num: usize, den: usize) -> Option<usize>;
}

pub struct Bar;

impl StrainRule for Bar {
	fn vec_len(&self) -> usize {
		1
	}

	fn dest_idx(&self, num: usize, den: usize) -> Option<usize> {
		match (num, den) {
			(0, 0) => Some(0),
			_ => None,
		}
	}
}

pub struct PlaneStrain;

impl StrainRule for PlaneStrain {
	fn vec_len(&self) -> usize {
		3
	}

	fn dest_idx(&self, num: usize, den: usize) -> Option<usize> {
		match (num, den) {
			(0, 0) => Some(0),
			(1, 1) => Some(1),
			(0, 1) => Some(2),
			(1, 0) => Some(2),
			_ => None,
		}
	}
}

pub struct ThreeDimensional;

impl StrainRule for ThreeDimensional {
	fn vec_len(&self) -> usize {
		6
	}

	fn dest_idx(&self, num: usize, den: usize) -> Option<usize> {
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