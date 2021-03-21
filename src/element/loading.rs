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
        match self {
            Constraint::PlainDof(x, y, z) => *[x, y, z][dim],
        }
    }
}
