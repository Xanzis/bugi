use crate::spatial::Point;

#[derive(Debug, Clone, Copy)]
pub enum Constraint {
	PlainDof(bool, bool, bool),
}

impl Constraint {
	pub fn is_plain(&self) -> bool {
		match self {
			Constraint::PlainDof(_, _, _) => true,
		}
	}

	pub fn plain_dim_struck(&self, dim: usize) -> bool {
		// return whether a constraint has removed a given degree of freedom
		if let Constraint::PlainDof(x, y, z) = self {
			*[x, y, z][dim]
		}
		else {
			panic!("non-plain constraint");
		}
	}
}