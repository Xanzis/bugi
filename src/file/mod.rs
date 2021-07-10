use crate::element::ElementAssemblage;
use crate::mesher::bounds::PlaneBoundary;
use std::error;
use std::fmt;
use std::fs;
use std::path::Path;

mod bbnd;
mod bmsh;

pub fn read_to_elas<P: AsRef<Path>>(path: P) -> Result<ElementAssemblage, FileError> {
    let ext = path
        .as_ref()
        .extension()
        .map_or(Err(FileError::NoExt), |e| Ok(e))?; // hideous one-liner :)
    let ext = ext.to_str().ok_or(FileError::NonUniPath)?;
    let ext = ext.to_string(); // cannot be bothered to deal with non-unicode paths

    let file = fs::read_to_string(path).or(Err(FileError::NoOpen))?;
    let lines = file.split("\n");

    match ext.as_str() {
        "bmsh" => bmsh::lines_to_elas(lines),
        _ => Err(FileError::BadType(ext)),
    }
}

pub fn read_to_bound<P: AsRef<Path>>(path: P) -> Result<PlaneBoundary, FileError> {
    let ext = path
        .as_ref()
        .extension()
        .map_or(Err(FileError::NoExt), |e| Ok(e))?;
    let ext = ext.to_str().ok_or(FileError::NonUniPath)?.to_string();

    let file = fs::read_to_string(path).or(Err(FileError::NoOpen))?;
    let lines = file.split("\n");

    match ext.as_str() {
        "bbnd" => bbnd::lines_to_bounds(lines),
        _ => Err(FileError::BadType(ext)),
    }
}

pub fn save_elas<P: AsRef<Path>>(path: P, elas: ElementAssemblage) {
    let serialized = bmsh::elas_to_bmsh(elas);

    fs::write(path, serialized).expect("could not write to file");
}

#[derive(Debug, Clone)]
pub enum FileError {
    NoExt,
    NoOpen,
    NonUniPath,
    BadParse(String),
    BadType(String),
}

impl error::Error for FileError {}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FileError: ")?;
        match self.clone() {
            FileError::NoExt => {
                write!(f, "cannot read file extension")
            }
            FileError::NoOpen => {
                write!(f, "cannot open file")
            }
            FileError::NonUniPath => {
                write!(f, "non-unicode path")
            }
            FileError::BadType(s) => {
                write!(f, "unsupported file type: {}", s)
            }
            FileError::BadParse(s) => {
                write!(f, "bad parse: {}", s)
            }
        }
    }
}

impl FileError {
    pub fn parse<T: ToString>(msg: T) -> Self {
        Self::BadParse(msg.to_string())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn load_test() {
        let elas_out = super::read_to_elas("example_files/square.bmsh");
        assert!(elas_out.is_ok());
        let mut elas = elas_out.unwrap();
        elas.calc_displacements();

        let mut vis = elas.visualize(50.0);
        vis.set_vals(elas.displacement_norms().unwrap());
        vis.draw("test_generated/disp_bmsh_square.png", ());
    }
}
