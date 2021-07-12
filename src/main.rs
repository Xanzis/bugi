#[macro_use]
extern crate clap;

use std::convert::From;
use std::error::Error;
use std::fmt;
use std::path;

use bugi::element;
use bugi::file;
use bugi::mesher::plane;
use bugi::visual::{color, VisOptions};

#[derive(Debug, Clone)]
enum BugiError {
    ArgError(String),
    RunError(String),
}

fn main() -> Result<(), BugiError> {
    let yaml = load_yaml!("cli.yml");
    let matches = clap::App::from_yaml(yaml).get_matches();

    match matches.subcommand() {
        ("linear", Some(linear_matches)) => linear(linear_matches),
        ("mesh", Some(mesh_matches)) => mesh(mesh_matches),
        ("", None) => Ok(()),
        _ => unreachable!(),
    }
}

fn linear<'a>(args: &clap::ArgMatches<'a>) -> Result<(), BugiError> {
    // compute a linear study with the given mesh/setup file
    let file_path = args.value_of("INPUT").unwrap();
    let file_path = path::Path::new(file_path);

    let mut elas = file::read_to_elas(file_path)?;

    elas.calc_displacements();

    let mut vis = elas.visualize(50.0);

    let node_vals = match args.value_of("nodevalue") {
        None | Some("displacement") => elas.displacement_norms().unwrap(),
        Some("vonmises") => elas.von_mises(),
        _ => return Err(BugiError::arg_error("unimplemented node value name")),
    };

    vis.set_vals(node_vals);

    let out_path = args.value_of("out").unwrap_or("out.png").to_string();

    let vis_options: VisOptions = match args.value_of("colormap") {
        None => ().into(),
        Some("hot") => VisOptions::with_color_map(Box::new(|x| color::hot_map(x))),
        Some("rgb") => VisOptions::with_color_map(Box::new(|x| color::rgb_map(x))),
        _ => return Err(BugiError::arg_error("unimplemented colormap name")),
    };

    // TODO allow manual specification of image size
    let vis_options = vis_options.with(vec!["im_size=2048"]);

    // TODO incorporate rust's PATH logic and check path validities

    vis.draw(out_path.as_str(), vis_options);

    Ok(())
}

fn mesh<'a>(args: &clap::ArgMatches<'a>) -> Result<(), BugiError> {
    // compute a mesh from a boundary definition
    // save a mesh definition file and a mesh visualization
    let file_path = args.value_of("INPUT").unwrap();
    let file_path = path::Path::new(file_path);

    let bnd = file::read_to_bound(file_path)?;

    // TODO will need to add 3d options when available
    let mut msh = plane::PlaneTriangulation::new(bnd);

    let size = args
        .value_of("elementsize")
        .unwrap()
        .parse::<f64>()
        .or(Err(BugiError::arg_error("could not parse element size")))?;

    match args.value_of("mesher") {
        None | Some("chew") => msh.chew_mesh(size),
        _ => {
            return Err(BugiError::arg_error(
                "unrecognised mesh algorithm specified",
            ))
        }
    }

    let vis_out_path = args.value_of("imageout").unwrap_or("out.png").to_string();
    let mesh_out_path = args.value_of("meshout").unwrap_or("out.bmsh").to_string();

    // TODO add rust Path logic to the save process

    let mut vis = msh.visualize();
    vis.draw(vis_out_path.as_str(), vec!["im_size=1024"]);

    let elas: element::ElementAssemblage = msh.assemble().map_err(|e| BugiError::run_error(e))?;

    file::save_elas(mesh_out_path.as_str(), elas);

    Ok(())
}

impl BugiError {
    fn arg_error<T: fmt::Display>(msg: T) -> Self {
        Self::ArgError(msg.to_string())
    }
    fn run_error<T: fmt::Display>(msg: T) -> Self {
        Self::RunError(msg.to_string())
    }
}

impl<T> From<T> for BugiError
where
    T: Error,
{
    fn from(x: T) -> Self {
        BugiError::run_error(x.to_string())
    }
}

impl fmt::Display for BugiError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BugiError: ")?;
        match self.clone() {
            BugiError::ArgError(msg) => {
                write!(f, "argument error: {}", msg)
            }
            BugiError::RunError(msg) => {
                write!(f, "runtime error: {}", msg)
            }
        }
    }
}
