use std::collections::{HashMap, HashSet};
use std::convert::From;
use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path;

use bugi::element;
use bugi::file;
use bugi::mesher::plane;
use bugi::visual::{color, VisOptions};

const VERSION: &str = "0.1.0";

#[derive(Debug, Clone)]
struct Args {
    command: String,
    args: Vec<String>,
    flags: HashSet<String>,
    named_vals: HashMap<String, String>,
}

#[derive(Debug, Clone)]
enum BugiError {
    NoArgs,
    ArgError(String),
    RunError(String),
}

fn main() -> Result<(), BugiError> {
    let args = env::args().skip(1);

    if args.len() == 0 {
        help();
        return Err(BugiError::NoArgs);
    }

    let args = Args::compile(args.collect()).map_err(|e| {
        help();
        e
    })?;

    if args.has_flag("help") {
        help();
        return Ok(());
    }

    if args.has_flag("version") {
        println!("bugi version {}", VERSION);
        return Ok(());
    }

    match args.command.as_str() {
        "linear" => linear(args),
        "mesh" => mesh(args),
        "" => {
            help();
            Err(BugiError::arg_error("no command supplied"))
        }
        _ => {
            help();
            Err(BugiError::arg_error("unrecognized command"))
        }
    }
}

fn linear(args: Args) -> Result<(), BugiError> {
    // compute a linear study with the given mesh/setup file
    let file_path = args
        .cmd_arg(0)
        .ok_or(BugiError::arg_error("missing path argument"))?;
    let file_path = path::Path::new(file_path.as_str());

    let mut elas = file::read_to_elas(file_path)?;

    elas.calc_displacements();

    let mut vis = elas.visualize(50.0);

    let node_vals = match args.arg_val("nodevalue") {
        None | Some("displacement") => elas.displacement_norms().unwrap(),
        Some("vonmises") => elas.von_mises(),
        _ => return Err(BugiError::arg_error("unimplemented node value name")),
    };

    vis.set_vals(node_vals);

    let out_path = match args.arg_val("out") {
        Some(s) => format!("{}.png", s),
        None => "out.png".to_string(),
    };

    let vis_options: VisOptions = match args.arg_val("colormap") {
        None => ().into(),
        Some("hot") => VisOptions::with_color_map(Box::new(|x| color::hot_map(x))),
        Some("rgb") => VisOptions::with_color_map(Box::new(|x| color::rgb_map(x))),
        _ => return Err(BugiError::arg_error("unimplemented colormap name")),
    };

    // TODO incorporate rust's PATH logic and check path validities

    vis.draw(out_path.as_str(), vis_options);

    Ok(())
}

fn mesh(args: Args) -> Result<(), BugiError> {
    // compute a mesh from a boundary definition
    // save a mesh definition file and a mesh visualization
    let file_path = args
        .cmd_arg(0)
        .ok_or(BugiError::arg_error("missing path argument"))?;
    let file_path = path::Path::new(file_path.as_str());

    let mut bnd = file::read_to_bound(file_path)?;

    // TODO will need to add 3d options when available
    let mut msh = plane::PlaneTriangulation::new(bnd);

    let size = match args.arg_val("elementsize") {
        None => Err(BugiError::arg_error("element size not specified")),
        Some(x) => x
            .parse::<f64>()
            .or(Err(BugiError::arg_error("could not parse element size"))),
    }?;

    match args.arg_val("mesher") {
        None | Some("chew") => msh.chew_mesh(size),
        _ => {
            return Err(BugiError::arg_error(
                "unrecognised mesh algorithm specified",
            ))
        }
    }

    let (vis_out_path, mesh_out_path) = match args.arg_val("out") {
        Some(s) => (format!("{}.png", s), format!("{}.bmsh", s)),
        None => ("out.png".to_string(), "out.bmsh".to_string()),
    };

    // TODO add rust Path logic to the save process

    let mut vis = msh.visualize();
    vis.draw(vis_out_path.as_str(), vec!["im_size=1024"]);

    let elas: element::ElementAssemblage = msh.assemble().map_err(|e| BugiError::run_error(e))?;

    file::save_elas(mesh_out_path.as_str(), elas);

    Ok(())
}

impl Args {
    fn compile(env_args: Vec<String>) -> Result<Self, BugiError> {
        let mut command = String::new();
        let mut args = Vec::new();
        let mut flags = HashSet::new();
        let mut named_vals = HashMap::new();
        let mut command_read = false;

        for mut s in env_args.into_iter() {
            if s.as_str().starts_with("--") {
                s.drain(..2);
                flags.insert(s);
            } else if s.as_str().starts_with("-") {
                s.drain(..1);
                let words: Vec<String> = s.as_str().split("=").map(|x| x.to_string()).collect();
                if words.len() != 2 {
                    return Err(BugiError::arg_error("bad key/value pair"));
                }
                named_vals.insert(words[0].clone(), words[1].clone());
            } else {
                if !command_read {
                    command = s;
                    command_read = true;
                } else {
                    args.push(s);
                }
            }
        }

        Ok(Args {
            command,
            args,
            flags,
            named_vals,
        })
    }

    fn cmd_arg(&self, i: usize) -> Option<String> {
        self.args.get(i).cloned()
    }

    fn arg_val(&self, key: &str) -> Option<&str> {
        self.named_vals.get(key).map(|x| x.as_str())
    }

    fn has_flag(&self, f: &str) -> bool {
        self.flags.contains(f)
    }
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
            BugiError::NoArgs => {
                write!(f, "no arguments supplied")
            }
            BugiError::RunError(msg) => {
                write!(f, "runtime error: {}", msg)
            }
        }
    }
}

const HELP: &str = "
NAME
	bugi - a finite element solver

SYNOPSIS
	bugi [--version] [--help] [-colormap=<name>]
		 <command> [<args>]

OPTIONS
	--version
		prints the bugi version

	--help
		prints this text

	-colormap=<name>
		sets the color map for graphic outputs

    -nodevalue=<value name>
        sets the node value type to visualize; default is displacement

    -elementsize=<number>
        sets the element size parameter for meshing operations

    -mesher=<mesh algorithm name>
        sets the mesh algorithm for meshing operations; default is chew

    -out=<name>
        sets the stem of the output file(s)

COMMANDS
	bugi linear <path>
		Run a linear finite element solver on the specified mesh/setup file.

    bugi mesh <path>
        Generate a mesh using the specified mesh file and element parameters
";

fn help() {
    eprintln!("{}", HELP);
}
