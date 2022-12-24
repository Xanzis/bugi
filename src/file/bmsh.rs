use std::convert::TryInto;

use crate::element::isopar;
use crate::element::loading::Constraint;
use crate::element::material::{Material, AL6061};
use crate::element::{ElementAssemblage, NodeId};
use crate::element::{ElementDescriptor, ElementType};
use crate::spatial::Point;

use nom::{self, IResult, Parser};

use super::FileError;

type NomStrErr<'a> = nom::error::Error<&'a str>;

fn parse_error(line_no: usize, text: &'static str) -> FileError {
    let s = format!("line {}: {}", line_no, text);
    FileError::BadParse(s)
}

fn code_to_desc(code: (u8, Vec<NodeId>)) -> ElementDescriptor {
    let tp = match code.0 {
        1 => unimplemented!("2-node bars temporarily unimplemented"),
        2 => ElementType::Isopar(isopar::ElementType::Triangle3),
        3 => unimplemented!("rectangles temporarily unimplemented"),
        _ => unimplemented!("unimplemented element type"),
    };

    ElementDescriptor::new(tp, code.1)
}

fn desc_to_code(desc: ElementDescriptor) -> (u8, Vec<NodeId>) {
    let desc = desc.into_parts();
    let tp_code = match desc.0 {
        ElementType::Isopar(itp) => match itp {
            isopar::ElementType::Triangle3 => 2,
        },
    };

    (tp_code, desc.1)
}

fn monotonic_list<'a, O, F>(mut f: F) -> impl FnMut(&'a str) -> IResult<&'a str, Vec<O>>
where
    F: Parser<&'a str, O, NomStrErr<'a>>,
    //    E: nom::error::ParseError<&'static str>
{
    // helper function for sequences of lines that should look like
    // count <x>\n0 <f>\n1 <f>\n...
    // finds 0 or more lines
    use nom::{bytes::complete::tag, character, error::ErrorKind, sequence::delimited};

    move |init_input| {
        let mut items = Vec::new();
        let (mut input, total_exp) =
            delimited(tag("count "), character::complete::u64, tag("\n"))(init_input.clone())?;
        let total_exp = total_exp as usize;

        // need any returned errors to contain the initial slice
        let error = nom::Err::Error(nom::error::Error::new(init_input, ErrorKind::Verify));

        for i in 0..total_exp {
            let (rem, num) = character::complete::u64(input)?;
            input = rem;

            if i != num as usize {
                return Err(error);
            }

            let (rem, _) = tag(" ")(input).map_err(|_: nom::Err<NomStrErr<'a>>| {
                nom::Err::Error(nom::error::Error::new(init_input, ErrorKind::Verify))
            })?;
            input = rem;
            let (rem, value) = f.parse(input).map_err(|_| {
                nom::Err::Error(nom::error::Error::new(init_input, ErrorKind::Verify))
            })?;
            input = rem;
            let (rem, _) = tag("\n")(input).map_err(|_: nom::Err<NomStrErr<'a>>| {
                nom::Err::Error(nom::error::Error::new(init_input, ErrorKind::Verify))
            })?;
            input = rem;

            items.push(value)
        }

        if items.len() == total_exp {
            Ok((input, items))
        } else {
            Err(error)
        }
    }
}

fn free_list<'a, O, F>(mut f: F) -> impl FnMut(&'a str) -> IResult<&'a str, Vec<O>>
where
    F: Parser<&'a str, O, NomStrErr<'a>>,
{
    // helper function for sequences of lines without item indices
    use nom::{bytes::complete::tag, character, error::ErrorKind, sequence::delimited};

    move |init_input| {
        let mut items = Vec::new();
        let (mut input, total_exp) =
            delimited(tag("count "), character::complete::u64, tag("\n"))(init_input.clone())?;
        let total_exp = total_exp as usize;

        // need any returned errors to contain the initial slice
        let error = nom::Err::Error(nom::error::Error::new(init_input, ErrorKind::Verify));

        for _ in 0..total_exp {
            // TODO mayyybe wrap an f.parse err in a new error
            let (rem, value) = f.parse(input).map_err(|_| {
                nom::Err::Error(nom::error::Error::new(init_input, ErrorKind::Verify))
            })?;
            input = rem;
            items.push(value);

            let (rem, _) = tag("\n")(input).map_err(|_: nom::Err<NomStrErr<'a>>| {
                nom::Err::Error(nom::error::Error::new(init_input, ErrorKind::Verify))
            })?;
            input = rem;
        }

        if items.len() == total_exp {
            Ok((input, items))
        } else {
            Err(error)
        }
    }
}

fn section_parse<'a, O, F>(
    sec_name: &'static str,
    f: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: Parser<&'a str, O, NomStrErr<'a>>,
{
    // helper for the bmsh section delimiters
    use nom::{bytes::complete::tag, sequence::delimited};

    delimited(
        delimited(tag("$"), tag(sec_name), tag("\n")),
        f,
        delimited(tag("$End"), tag(sec_name), tag("\n")),
    )
}

fn named_value<'a, O, F>(name: &'static str, f: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: Parser<&'a str, O, NomStrErr<'a>>,
    //    E: nom::error::ParseError<&'static str>
{
    // helper for single line named values
    use nom::{
        bytes::complete::tag,
        sequence::{delimited, terminated},
    };

    delimited(terminated(tag(name), tag(" ")), f, tag("\n"))
}

struct ParseHeader {
    version: u8,
    dim: usize,
    material: Material,
    thickness: Option<f64>,
    area: Option<f64>,
}

impl ParseHeader {
    fn into_elas(self) -> ElementAssemblage {
        let mut res = ElementAssemblage::new(self.dim, self.material);
        if let Some(t) = self.thickness {
            res.set_thickness(t);
        }
        if let Some(a) = self.area {
            res.set_area(a);
        }
        res
    }
}

pub fn bmsh_to_elas(file: &str) -> Result<ElementAssemblage, FileError> {
    use nom::{
        bytes::complete::tag,
        character::complete::{self, alphanumeric1},
        combinator::{map, opt},
        multi::separated_list0,
        number::complete::double,
        sequence::{separated_pair, terminated, tuple},
    };

    // preprocess to remove comment and empty lines
    // could replace with comment / blank aware parser
    let clean_file: String = file
        .split("\n")
        .filter(|l| !l.is_empty() && !l.starts_with("#"))
        .fold(String::new(), |mut s, x| {
            s.push_str(x);
            s.push_str("\n");
            s
        });
    println!("{:?}", clean_file);
    let input = clean_file.as_str();

    // parse the header
    let version = named_value("version", complete::u8);
    let dim = named_value("dim", complete::u64);
    let material = named_value("material", alphanumeric1);
    let thickness = named_value("thickness", double);
    let area = named_value("area", double);

    let (input, header) = section_parse(
        "Header",
        map(
            tuple((version, dim, material, opt(thickness), opt(area))),
            |(v, d, m, t, a)| ParseHeader {
                version: v,
                dim: d as usize,
                material: m.parse().expect("bad material"),
                thickness: t,
                area: a,
            },
        ),
    )(input)?;

    if header.version != 1 {
        return Err(FileError::parse(format!(
            "Unsupported version: {}",
            header.version
        )));
    }

    println!("parsed header. input:\n{:?}", input);

    // parse the remaining segments
    let (input, nodes): (_, Vec<Point>) = section_parse(
        "Nodes",
        monotonic_list(map(separated_list0(tag("/"), double), |x| {
            x.try_into().expect("bad node position string")
        })),
    )(input)?;

    println!("parsed nodes. input:\n{:?}", input);

    let (input, elements): (_, Vec<(u8, Vec<usize>)>) = section_parse(
        "Elements",
        monotonic_list(separated_pair(
            complete::u8,
            tag(" "),
            separated_list0(tag("/"), map(complete::u64, |x| x as usize)),
        )),
    )(input)?;

    println!("parsed elements. input:\n{:?}", input);

    let (input, constraints): (_, Vec<(usize, Constraint)>) = section_parse(
        "Constraints",
        free_list(separated_pair(
            map(complete::u64, |x| x as usize),
            tag(" "),
            map(
                separated_pair(
                    complete::u8,
                    tag(" "),
                    separated_list0(tag("/"), map(complete::u64, |x| x as usize)),
                ),
                // TODO: remove redundant info from make_constraint
                |(tp, dims)| make_constraint(0, tp, dims).expect("bad constraint"),
            ),
        )),
    )(input)?;

    println!("parsed constraints. input:\n{:?}", input);

    let (input, forces): (_, Vec<(usize, Point)>) = section_parse(
        "Forces",
        free_list(separated_pair(
            map(complete::u64, |x| x as usize),
            tag(" "),
            map(separated_list0(tag("/"), double), |x| {
                x.try_into().expect("bad force vector string")
            }),
        )),
    )(input)?;

    println!("parsed forces. input:\n{:?}", input);

    let (_input, dist_forces): (_, Vec<(usize, usize, Point)>) = section_parse(
        "DistForces",
        free_list(tuple((
            terminated(map(complete::u64, |x| x as usize), tag(" ")),
            terminated(map(complete::u64, |x| x as usize), tag(" ")),
            map(separated_list0(tag("/"), double), |x| {
                x.try_into().expect("bad force vector string")
            }),
        ))),
    )(input)?;

    // done with parsing, time to assemble the output
    let mut elas = header.into_elas();

    let node_ids = elas.add_nodes(&nodes);

    // TODO write an elas API to generate an element with the suggested type
    for (tp, ns) in elements.into_iter() {
        let nids = ns.into_iter().map(|i| node_ids[i]).collect();
        let desc = code_to_desc((tp, nids));
        elas.add_element(desc);
    }

    for (n, con) in constraints.into_iter() {
        elas.add_constraint(node_ids[n], con);
    }

    for (n, frc) in forces.into_iter() {
        elas.add_conc_force(node_ids[n], frc);
    }

    for (na, nb, frc) in dist_forces.into_iter() {
        elas.add_dist_line_force(node_ids[na], node_ids[nb], frc);
    }

    Ok(elas)
}

pub fn elas_to_bmsh(elas: ElementAssemblage) -> String {
    // construct a bmsh representation of the elas for dumping to file
    // assumes elas is properly constructed and panics otherwise

    let dim = elas.dim();

    let mut res = "$Header\nversion 1".to_string();
    res += &format!("\ndim {}", dim);
    res += &format!("\nmaterial {}", elas.material().to_string());
    if let Some(t) = elas.thickness() {
        res += &format!("\nthickness {}", t);
    }
    if let Some(a) = elas.area() {
        res += &format!("\narea {}", a);
    }
    res += "\n$EndHeader\n\n$Nodes";

    let nodes = elas.nodes();
    res += &format!("\ncount {}", nodes.len());
    for (i, n) in nodes.into_iter().enumerate() {
        match dim {
            1 => res += &format!("\n{} {:.6}", i, n[0]),
            2 => res += &format!("\n{} {:.6}/{:.6}", i, n[0], n[1]),
            3 => res += &format!("\n{} {:.6}/{:.6}/{:.6}", i, n[0], n[1], n[2]),
            _ => unreachable!(),
        }
    }
    res += "\n$EndNodes\n\n$Elements";

    let descs = elas.element_descriptors();
    res += &format!("\ncount {}", descs.len());
    for (i, desc) in descs.into_iter().enumerate() {
        let (type_id, node_ids) = desc_to_code(desc);
        // element always has at least one node
        let mut to_write = format!("\n{} {} {}", i, type_id, node_ids[0].into_idx());

        to_write.extend(
            node_ids
                .iter()
                .skip(1)
                .map(|nid| format!("/{}", nid.into_idx())),
        );

        to_write.push_str("\n");
        res += &to_write;
    }
    res += "\n$EndElements\n\n$Constraints";

    let cons = elas.constraints();
    res += &format!("\ncount {}", cons.len());
    for (n, con) in cons {
        // TODO update the '0' when more constraint types are available
        res += &format!(
            "\n{} 0 {}",
            n.into_idx(),
            if con.plain_dim_struck(0) { 1 } else { 0 }
        );

        // TODO fix constraint reader to accept 2 and 1-dim assemblage constraints
        for i in 1..3 {
            res += if con.plain_dim_struck(i) { "/1" } else { "/0" };
        }
    }
    res += "\n$EndConstraints\n\n$Forces";

    let forces = elas.conc_forces();
    res += &format!("\ncount {}", forces.len());
    for (n, frc) in forces {
        res += &format!("\n{} {:.3}", n.into_idx(), frc[0]);

        for i in (0..dim).skip(1) {
            res += &format!("/{:.3}", frc[i]);
        }
    }
    res.push_str("\n$EndForces\n\n$DistForces");

    let dist_forces = elas.dist_forces();
    res += &format!("\ncount {}", dist_forces.len());
    for ((na, nb), frc) in dist_forces {
        res += &format!("\n{} {} {:.3}", na.into_idx(), nb.into_idx(), frc[0]);

        for i in (0..dim).skip(1) {
            res += &format!("/{:.3}", frc[i]);
        }
    }
    res += "\n$EndDistForces";

    res
}

fn make_constraint(no: usize, tp: u8, dims: Vec<usize>) -> Result<Constraint, FileError> {
    // simple for now
    match tp {
        0 => {
            if dims.len() != 3 {
                Err(parse_error(
                    no,
                    "constraint requires three dimensional specifiers",
                ))
            } else {
                let (a, b, c) = (dims[0] != 0, dims[1] != 0, dims[2] != 0);
                Ok(Constraint::PlainDof(a, b, c))
            }
        }
        _ => Err(parse_error(no, "unsupported constraint type")),
    }
}

// BMSH file specification
// Element type numberings intended to be compatible with gmsh equivalents
// Anything in <> should be replaced (<> as well) with the specified contents
//
// ~~file begin~~
//
// $Header
// version <ASCII uint, currently 1>
// dim <ASCII 1, 2, or 3, nodes must have matching numbers of values>
// material <material name>
// # include the rest of the header lines only if necessary
// thickness <ASCII float>
// area <ASCII float>
// $EndHeader
//
// # blank lines or those beginning with '#' are ignored
// $Nodes
// count <ASCII uint, number of nodes>
// <node id, must currently be sequential from 0> <ASCII float x val>/<y val if applicable>/<z val>
// ...
// $EndNodes
// $Elements
// count <ASCII uint, number of elements>
// <element id, sequential from 0> <ASCII uint element type> <node id 1>/<node id 2> ...
// ...
// $EndElements
// $Constraints
// count <ASCII unit, number of constraints>
// <node id> <constraint type> <data 1 if applicable (see constraint types)>/<data 2>/<data 3>
// ...
// $EndConstraints
// $Forces
// count <ASCII uint, number of forces>
// <node id> <ASCII float x val>/<y val if applicable>/<z val if applicable>
// ...
// $EndForces
// $DistForces
// count <ASCII uint, number of forces>
// <node id a> <node id b> <ASCII float x val>/<y val if applicable>/<z val if applicable>
// ...
// $EndDistForces
// ~~file end~~
//
// Element types: (only showing those so far implemented)
// 1: 2-node line
// 2: 3-node triangle
// 3: 4-node quadrangle
