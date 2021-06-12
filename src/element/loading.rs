use std::str::FromStr;

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

impl FromStr for Constraint {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // parse as a string like xy, xz, y, where presence of a dim means it is struck
        let mut x = false;
        let mut y = false;
        let mut z = false;
        for c in s.chars() {
            match c {
                'x' => x = true,
                'y' => y = true,
                'z' => z = true,
                _ => return Err(()),
            }
        }

        Ok(Constraint::PlainDof(x, y, z))
    }
}