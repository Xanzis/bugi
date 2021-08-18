#[macro_use]
extern crate clap;

use std::convert::From;
use std::error::Error;
use std::fmt;
use std::path;

use bugi::element;
use bugi::file;
use bugi::matrix::solve::direct::{CholeskyEnvelopeSolver, DenseGaussSolver};
use bugi::matrix::solve::iterative::GaussSeidelSolver;
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
        ("modal", Some(modal_matches)) => modal(modal_matches),
        ("", None) => Ok(()),
        _ => unreachable!(),
    }
}

fn linear<'a>(args: &clap::ArgMatches<'a>) -> Result<(), BugiError> {
    // compute a linear study with the given mesh/setup file
    eprintln!("computing linear study ...");

    let file_path = args.value_of("INPUT").unwrap();
    let file_path = path::Path::new(file_path);

    eprintln!("reading mesh file ...");
    let mut elas = file::read_to_elas(file_path)?;
    eprintln!("file read (node count: {})", elas.node_count());

    eprintln!("solving ...");

    // compute deformation given load parameters
    let dfm = match args.value_of("solver") {
        None => elas.calc_displacements::<DenseGaussSolver>(),
        Some("densegauss") => elas.calc_displacements::<DenseGaussSolver>(),
        Some("cholesky") => elas.calc_displacements::<CholeskyEnvelopeSolver>(),
        Some("gaussseidel") => elas.calc_displacements::<GaussSeidelSolver>(),
        _ => return Err(BugiError::arg_error("unimplemented solver name")),
    };
    eprintln!("solution complete\npost processing solution ...");

    let max_von_mises = dfm
        .von_mises()
        .into_iter()
        .max_by(|x, y| x.partial_cmp(y).expect("encountered unstable stress value"))
        .unwrap();
    let max_displacement = dfm
        .displacement_norms()
        .into_iter()
        .max_by(|x, y| x.partial_cmp(y).expect("encountered unstable stress value"))
        .unwrap();

    let scale = args
        .value_of("scale")
        .unwrap_or("50.0")
        .parse::<f64>()
        .map_err(|_| BugiError::arg_error("could not parse displacement scale argument"))?;
    let mut vis = dfm.visualize(scale);

    let node_vals = match args.value_of("nodevalue") {
        None | Some("displacement") => dfm.displacement_norms(),
        Some("vonmises") => dfm.von_mises(),
        _ => return Err(BugiError::arg_error("unimplemented node value name")),
    };

    vis.set_vals(node_vals);

    let out_path = args.value_of("out").unwrap_or("out.png").to_string();

    let vis_options = VisOptions::new();

    let vis_options = match args.value_of("colormap") {
        None => ().into(),
        Some("hot") => vis_options.color_map(Box::new(|x| color::hot_map(x))),
        Some("rgb") => vis_options.color_map(Box::new(|x| color::rgb_map(x))),
        _ => return Err(BugiError::arg_error("unimplemented colormap name")),
    };

    let im_size = args
        .value_of("imsize")
        .unwrap_or("1024")
        .parse::<u32>()
        .map_err(|_| BugiError::arg_error("could not parse image size argument"))?;
    let vis_options = vis_options.im_size(im_size);

    let show_mesh = args
        .value_of("showmesh")
        .unwrap_or("true")
        .parse::<bool>()
        .map_err(|_| BugiError::arg_error("could not parse show mesh argument"))?;
    let vis_options = vis_options.show_mesh(show_mesh);

    // TODO incorporate rust's PATH logic and check path validities

    vis.draw(out_path.as_str(), vis_options);

    eprintln!("processing complete\n---------------");

    println!(
        "max von mises: {}\nmax displacement: {}",
        max_von_mises, max_displacement
    );

    Ok(())
}

fn mesh<'a>(args: &clap::ArgMatches<'a>) -> Result<(), BugiError> {
    // compute a mesh from a boundary definition
    // save a mesh definition file and a mesh visualization
    eprintln!("computing mesh ...");

    let file_path = args.value_of("INPUT").unwrap();
    let file_path = path::Path::new(file_path);

    eprintln!("initializing boundary ...");
    let bnd = file::read_to_bound(file_path)?;

    // TODO will need to add 3d options when available
    let mut msh = plane::PlaneTriangulation::new(bnd);
    eprintln!("boundary initialized");

    let size = args
        .value_of("elementsize")
        .unwrap()
        .parse::<f64>()
        .or(Err(BugiError::arg_error("could not parse element size")))?;

    eprintln!("meshing ...");
    match args.value_of("mesher") {
        None | Some("chew") => msh.chew_mesh(size),
        _ => {
            return Err(BugiError::arg_error(
                "unrecognised mesh algorithm specified",
            ))
        }
    }

    eprintln!("mesh complete\nvisualizing ...");

    let vis_out_path = args.value_of("imageout").unwrap_or("out.png").to_string();
    let mesh_out_path = args.value_of("meshout").unwrap_or("out.bmsh").to_string();

    // TODO add rust Path logic to the save process

    let mut vis = msh.visualize();

    let im_size: u32 = args
        .value_of("imsize")
        .unwrap_or("1024")
        .parse()
        .map_err(|_| BugiError::arg_error("could not parse image size argument"))?;

    let vis_options = VisOptions::new().im_size(im_size);

    vis.draw(vis_out_path.as_str(), vis_options);

    eprintln!("visualization complete\nbuilding element assemblage ...");

    let elas: element::ElementAssemblage = msh.assemble().map_err(|e| BugiError::run_error(e))?;

    file::save_elas(mesh_out_path.as_str(), elas);

    eprintln!("mesh generation complete");

    Ok(())
}

fn modal<'a>(args: &clap::ArgMatches<'a>) -> Result<(), BugiError> {
    // compute a linear study with the given mesh/setup file
    eprintln!("computing modal study ...");

    let file_path = args.value_of("INPUT").unwrap();
    let file_path = path::Path::new(file_path);

    eprintln!("reading mesh file ...");
    let mut elas = file::read_to_elas(file_path)?;
    eprintln!("file read (node count: {})", elas.node_count());

    let num: usize = args
        .value_of("NUM")
        .unwrap()
        .parse()
        .map_err(|_| BugiError::arg_error("could not parse mode count argument"))?;

    eprintln!("solving ...");
    let dfms = elas.calc_modes(num);
    eprintln!("solution complete\npost processing solution ...");

    let scale: f64 = args
        .value_of("scale")
        .unwrap_or("50.0")
        .parse()
        .map_err(|_| BugiError::arg_error("could not parse displacement scale argument"))?;

    for (i, dfm) in dfms.iter().enumerate() {
        let mut vis = dfm.visualize(scale);
        vis.set_vals(dfm.displacement_norms());

        let vis_options = VisOptions::new();

        // recomputing the vis_options each time is pretty dumb, but ok hack for now
        // problem is that VisOptions is not Clone

        let vis_options = match args.value_of("colormap") {
            None => ().into(),
            Some("hot") => vis_options.color_map(Box::new(|x| color::hot_map(x))),
            Some("rgb") => vis_options.color_map(Box::new(|x| color::rgb_map(x))),
            _ => return Err(BugiError::arg_error("unimplemented colormap name")),
        };

        let im_size = args
            .value_of("imsize")
            .unwrap_or("1024")
            .parse::<u32>()
            .map_err(|_| BugiError::arg_error("could not parse image size argument"))?;
        let vis_options = vis_options.im_size(im_size);

        let show_mesh = args
            .value_of("showmesh")
            .unwrap_or("true")
            .parse::<bool>()
            .map_err(|_| BugiError::arg_error("could not parse show mesh argument"))?;
        let vis_options = vis_options.show_mesh(show_mesh);

        let out_path_full = format!("out_{:02}.png", i);

        vis.draw(out_path_full.as_str(), vis_options);

        println!(
            "mode {:02} frequency (rad/s): {:.5}",
            i,
            dfm.frequency().unwrap()
        );
    }

    eprintln!("processing complete\n---------------");

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
