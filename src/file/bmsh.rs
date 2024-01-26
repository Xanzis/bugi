use std::convert::TryInto;

use crate::element::constraint::Constraint;
use crate::element::material::Material;
use crate::element::strain::Condition;
use crate::element::ElementDescriptor;
use crate::element::{ElementAssemblage, NodeId};
use crate::spatial::Point;

use nom::{self, IResult, Parser};

use super::FileError;

type NomStrErr<'a> = nom::error::Error<&'a str>;

fn parse_error(line_no: usize, text: &'static str) -> FileError {
    let s = format!("line {}: {}", line_no, text);
    FileError::BadParse(s)
}

fn code_to_desc(code: Vec<NodeId>) -> ElementDescriptor {
    assert!(code.len() == 3);

    ElementDescriptor::new([code[0], code[1], code[2]])
}

fn desc_to_code(desc: ElementDescriptor) -> [NodeId; 3] {
    desc.into_parts()
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
            delimited(tag("count "), character::complete::u64, tag("\n"))(init_input)?;
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
            delimited(tag("count "), character::complete::u64, tag("\n"))(init_input)?;
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
    material: Material,
    condition: Condition,
}

impl ParseHeader {
    fn into_elas(self) -> ElementAssemblage {
        ElementAssemblage::new(self.material, self.condition)
    }
}

pub fn bmsh_to_elas(file: &str) -> Result<ElementAssemblage, FileError> {
    use nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{self, alphanumeric1, not_line_ending},
        combinator::map,
        multi::separated_list0,
        number::complete::double,
        sequence::{preceded, separated_pair, terminated, tuple},
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
    let input = clean_file.as_str();

    // parse the header
    let version = named_value("version", complete::u8);
    let material = named_value("material", alphanumeric1);

    let condition_parser = alt((
        map(preceded(tag("planestrain "), double), |t: f64| {
            Condition::PlaneStrain(t)
        }),
        map(preceded(tag("planestress "), double), |t: f64| {
            Condition::PlaneStress(t)
        }),
        map(tag("axisymmetric"), |_| Condition::Axisymmetric),
    ));

    let condition = named_value("condition", condition_parser);

    let (input, header) = section_parse(
        "Header",
        map(tuple((version, material, condition)), |(v, m, c)| {
            ParseHeader {
                version: v,
                material: m.parse().expect("bad material"),
                condition: c,
            }
        }),
    )(input)?;

    if header.version != 1 {
        return Err(FileError::parse(format!(
            "Unsupported version: {}",
            header.version
        )));
    }

    // parse the remaining segments
    let (input, nodes): (_, Vec<Point>) = section_parse(
        "Nodes",
        monotonic_list(map(separated_list0(tag("/"), double), |x| {
            x.try_into().expect("bad node position string")
        })),
    )(input)?;

    let (input, elements): (_, Vec<Vec<usize>>) = section_parse(
        "Elements",
        monotonic_list(separated_list0(
            tag("/"),
            map(complete::u64, |x| x as usize),
        )),
    )(input)?;

    let (input, constraints): (_, Vec<(usize, Constraint)>) = section_parse(
        "Constraints",
        free_list(separated_pair(
            map(complete::u64, |x| x as usize),
            tag(" "),
            map(
                not_line_ending,
                // TODO: remove redundant info from make_constraint
                |con: &str| con.parse::<Constraint>().expect("bad constraint"),
            ),
        )),
    )(input)?;

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
    for ns in elements.into_iter() {
        let nids = ns.into_iter().map(|i| node_ids[i]).collect();
        let desc = code_to_desc(nids);
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

    let mut res = "$Header\nversion 1".to_string();
    res += &format!("\nmaterial {}", elas.material().to_string());
    res += &format!("\ncondition {}", elas.condition().to_string());
    res += "\n$EndHeader\n\n$Nodes";

    let nodes = elas.nodes();
    res += &format!("\ncount {}", nodes.len());
    for (i, n) in nodes.into_iter().enumerate() {
        res += &format!("\n{} {:.6}/{:.6}", i, n[0], n[1]);
    }
    res += "\n$EndNodes\n\n$Elements";

    let descs = elas.element_descriptors();
    res += &format!("\ncount {}", descs.len());
    for (i, desc) in descs.into_iter().enumerate() {
        let node_ids = desc_to_code(desc);
        // element always has at least one node
        let mut to_write = format!("\n{} {}", i, node_ids[0].into_idx());

        to_write.extend(
            node_ids
                .iter()
                .skip(1)
                .map(|nid| format!("/{}", nid.into_idx())),
        );

        res += &to_write;
    }
    res += "\n$EndElements\n\n$Constraints";

    let cons = elas.constraints();
    let cons = cons
        .into_iter()
        .filter(|(n, con)| *con != Constraint::Free)
        .collect::<Vec<_>>();
    res += &format!("\ncount {}", cons.len());
    let mut i = 0;
    for (n, con) in cons {
        // TODO update the '0' when more constraint types are available
        res += &format!("\n{} {}", n.into_idx(), con.to_string(),);
    }
    res += "\n$EndConstraints\n\n$Forces";

    let forces = elas.conc_forces();
    res += &format!("\ncount {}", forces.len());
    for (n, frc) in forces {
        res += &format!("\n{} {:.3}/{:.3}", n.into_idx(), frc[0], frc[1]);
    }
    res.push_str("\n$EndForces\n\n$DistForces");

    let dist_forces = elas.dist_forces();
    res += &format!("\ncount {}", dist_forces.len());
    for ((na, nb), frc) in dist_forces {
        res += &format!(
            "\n{} {} {:.3}/{:.3}",
            na.into_idx(),
            nb.into_idx(),
            frc[0],
            frc[1]
        );
    }
    res += "\n$EndDistForces";

    res
}

fn make_constraint(no: usize, tp: u8, dims: Vec<usize>) -> Result<Constraint, FileError> {
    // TODO update file format to reflect new constraint logic
    match tp {
        0 => {
            if dims.len() != 3 {
                Err(parse_error(
                    no,
                    "constraint requires three dimensional specifiers",
                ))
            } else {
                let (a, b, c) = (dims[0] != 0, dims[1] != 0, dims[2] != 0);
                if a && b {
                    Ok(Constraint::XYFixed)
                } else if a {
                    Ok(Constraint::XFixed)
                } else if b {
                    Ok(Constraint::YFixed)
                } else {
                    Err(parse_error(no, "aaa"))
                }
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
