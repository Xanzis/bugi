use std::convert::TryInto;

use crate::element::isopar::ElementType;
use crate::element::loading::Constraint;
use crate::element::material::{Material, AL6061};
use crate::element::ElementAssemblage;
use crate::spatial::Point;

use super::FileError;

#[derive(Debug, Clone)]
enum ReadState {
    Start,
    Header { idx: usize },
    NodeStart,
    Nodes { in_body: bool },
    ElementStart,
    Elements { in_body: bool },
    ConstraintStart,
    Constraints { in_body: bool },
    ForceStart,
    Forces { in_body: bool },
    DistForceStart,
    DistForces { in_body: bool },
    End,
}

fn parse_error(line_no: usize, text: &'static str) -> FileError {
    let s = format!("line {}: {}", line_no, text);
    FileError::BadParse(s)
}

pub fn lines_to_elas<'a, T: Iterator<Item = &'a str>>(
    mut lines: T,
) -> Result<ElementAssemblage, FileError> {
    let mut no = 0;

    let mut version: Option<u8> = None;
    let mut dim: Option<usize> = None;
    let mut material: Option<Material> = None;
    let mut thickness: Option<f64> = None;
    let mut area: Option<f64> = None;
    let mut node_count = 0;
    let mut nodes: Vec<Point> = Vec::new();
    let mut el_count = 0;
    let mut elements: Vec<(u8, Vec<usize>)> = Vec::new();
    let mut cons_count = 0;
    let mut constraints: Vec<(usize, Constraint)> = Vec::new();
    let mut force_count = 0;
    let mut forces: Vec<(usize, Point)> = Vec::new();
    let mut dist_force_count = 0;
    let mut dist_forces: Vec<(usize, usize, Point)> = Vec::new();

    let mut state = ReadState::Start;

    while let Some(line) = lines.next() {
        no += 1;
        if line.is_empty() || line.starts_with("#") {
            continue;
        }
        state = match state {
            ReadState::Start => {
                if line == "$Header" {
                    ReadState::Header { idx: 0 }
                } else {
                    return Err(parse_error(no, "expected $Header"));
                }
            }
            ReadState::Header { idx } => match idx {
                0 => {
                    let (n, v) = name_and_num(no, line)?;
                    if n != "version" {
                        return Err(parse_error(no, "expected version"));
                    }
                    version = Some(v as u8);
                    ReadState::Header { idx: 1 }
                }
                1 => {
                    let (n, v) = name_and_num(no, line)?;
                    if n != "dim" {
                        return Err(parse_error(no, "expected dimension"));
                    }
                    dim = Some(v);
                    ReadState::Header { idx: 2 }
                }
                2 => {
                    let (n, m) = name_and_name(no, line)?;
                    if n != "material" {
                        return Err(parse_error(no, "expected material"));
                    }
                    material = Some(make_material(no, m)?);
                    ReadState::Header { idx: 3 }
                }
                _ => {
                    if line == "$EndHeader" {
                        ReadState::NodeStart
                    } else {
                        if line.starts_with("thickness") {
                            thickness = Some(name_and_float(no, line)?.1);
                        } else if line.starts_with("area") {
                            area = Some(name_and_float(no, line)?.1);
                        } else {
                            return Err(parse_error(no, "unrecognized header field"));
                        }
                        ReadState::Header { idx: 3 }
                    }
                }
            },
            ReadState::NodeStart => {
                if line != "$Nodes" {
                    return Err(parse_error(no, "expected $Nodes"));
                }
                ReadState::Nodes { in_body: false }
            }
            ReadState::Nodes { in_body } => match in_body {
                false => {
                    let (n, v) = name_and_num(no, line)?;
                    if n != "count" {
                        return Err(parse_error(no, "expected count"));
                    }
                    node_count = v;
                    ReadState::Nodes { in_body: true }
                }
                true => {
                    if line == "$EndNodes" {
                        if nodes.len() != node_count {
                            return Err(parse_error(no, "did not read specified node count"));
                        }
                        ReadState::ElementStart
                    } else {
                        let (id, p) = id_and_point(no, line)?;
                        if id != nodes.len() {
                            return Err(parse_error(no, "out-of-order node id"));
                        }
                        nodes.push(p);
                        ReadState::Nodes { in_body: true }
                    }
                }
            },
            ReadState::ElementStart => {
                if line != "$Elements" {
                    return Err(parse_error(no, "expected $Elements"));
                }
                ReadState::Elements { in_body: false }
            }
            ReadState::Elements { in_body } => match in_body {
                false => {
                    let (n, v) = name_and_num(no, line)?;
                    if n != "count" {
                        return Err(parse_error(no, "expected count"));
                    }
                    el_count = v;
                    ReadState::Elements { in_body: true }
                }
                true => {
                    if line == "$EndElements" {
                        if elements.len() != el_count {
                            return Err(parse_error(no, "did not read specified element count"));
                        }
                        ReadState::ConstraintStart
                    } else {
                        let (id, tp, ns) = id_type_and_list(no, line)?;
                        if id != elements.len() {
                            return Err(parse_error(no, "out-of-order element id"));
                        }
                        elements.push((tp, ns));
                        ReadState::Elements { in_body: true }
                    }
                }
            },
            ReadState::ConstraintStart => {
                if line != "$Constraints" {
                    return Err(parse_error(no, "expected $Constraints"));
                }
                ReadState::Constraints { in_body: false }
            }
            ReadState::Constraints { in_body } => match in_body {
                false => {
                    let (n, v) = name_and_num(no, line)?;
                    if n != "count" {
                        return Err(parse_error(no, "expected count"));
                    }
                    cons_count = v;
                    ReadState::Constraints { in_body: true }
                }
                true => {
                    if line == "$EndConstraints" {
                        if constraints.len() != cons_count {
                            return Err(parse_error(no, "did not read specified constraint count"));
                        }
                        ReadState::ForceStart
                    } else {
                        let (id, tp, ds) = id_type_and_list(no, line)?;
                        if id >= nodes.len() {
                            return Err(parse_error(
                                no,
                                "constraint refers to out-of-bounds node index",
                            ));
                        }
                        constraints.push((id, make_constraint(no, tp, ds)?));
                        ReadState::Constraints { in_body: true }
                    }
                }
            },
            ReadState::ForceStart => {
                if line != "$Forces" {
                    return Err(parse_error(no, "expected $Forces"));
                }
                ReadState::Forces { in_body: false }
            }
            ReadState::Forces { in_body } => match in_body {
                false => {
                    let (n, v) = name_and_num(no, line)?;
                    if n != "count" {
                        return Err(parse_error(no, "expected count"));
                    }
                    force_count = v;
                    ReadState::Forces { in_body: true }
                }
                true => {
                    if line == "$EndForces" {
                        if forces.len() != force_count {
                            return Err(parse_error(no, "did not read specified force count"));
                        }
                        ReadState::DistForceStart
                    } else {
                        let (id, p) = id_and_point(no, line)?;
                        if id >= nodes.len() {
                            return Err(parse_error(
                                no,
                                "force refers to out-of-bounds node index",
                            ));
                        }
                        forces.push((id, p));
                        ReadState::Forces { in_body: true }
                    }
                }
            },
            ReadState::DistForceStart => {
                if line != "$DistForces" {
                    return Err(parse_error(no, "expected $DistForces"));
                }
                ReadState::DistForces { in_body: false }
            }
            ReadState::DistForces { in_body } => match in_body {
                false => {
                    let (n, v) = name_and_num(no, line)?;
                    if n != "count" {
                        return Err(parse_error(no, "expected count"));
                    }
                    dist_force_count = v;
                    ReadState::DistForces { in_body: true }
                }
                true => {
                    if line == "$EndDistForces" {
                        if dist_forces.len() != dist_force_count {
                            return Err(parse_error(
                                no,
                                "did not read specified distributed force count",
                            ));
                        }
                        ReadState::End
                    } else {
                        let (ida, idb, p) = id_id_and_point(no, line)?;
                        if ida >= nodes.len() || idb >= nodes.len() {
                            return Err(parse_error(
                                no,
                                "force refers to out-of-bounds node index",
                            ));
                        }
                        dist_forces.push((ida, idb, p));
                        ReadState::DistForces { in_body: true }
                    }
                }
            },
            ReadState::End => return Err(parse_error(no, "unexpected line at end of file")),
        };
    }

    if version != Some(1) {
        return Err(parse_error(no, "unsupported version"));
    }

    // done with parsing, time to assemble the output
    // unwraps should be ok here - failures to set should all be caught earlier
    let mut elas = ElementAssemblage::new(dim.unwrap(), material.unwrap());

    if let Some(t) = thickness {
        elas.set_thickness(t);
    }
    if let Some(a) = area {
        elas.set_area(a);
    }

    elas.add_nodes(nodes);

    // TODO write an elas API to generate an element with the suggested type
    for (_, ns) in elements.into_iter() {
        elas.add_element(ns);
    }

    for (n, con) in constraints.into_iter() {
        elas.add_constraint(n, con);
    }

    for (n, frc) in forces.into_iter() {
        elas.add_conc_force(n, frc);
    }

    for (na, nb, frc) in dist_forces.into_iter() {
        elas.add_dist_line_force(na, nb, frc);
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

    let e_node_idxs = elas.element_node_idxs();
    let e_types = elas.element_types();
    res += &format!("\ncount {}", e_types.len());
    for (i, (en, et)) in e_node_idxs.into_iter().zip(e_types.into_iter()).enumerate() {
        res += &format!("\n{} ", i);

        let type_id = match et {
            ElementType::Bar2Node => 1,
            ElementType::Triangle3Node => 2,
            _ => panic!("unsupported element type for bmsh output"),
        };

        res += &format!("{} {}", type_id, en.get(0).expect("no 0 node elements"));
        for n in en.into_iter().skip(1) {
            res += &format!("/{}", n);
        }
    }
    res += "\n$EndElements\n\n$Constraints";

    let cons = elas.constraints();
    res += &format!("\ncount {}", cons.len());
    for (n, con) in cons {
        // TODO update the '0' when more constraint types are available
        res += &format!("\n{} 0 {}", n, if con.plain_dim_struck(0) { 1 } else { 0 });

        // TODO fix constraint reader to accept 2 and 1-dim assemblage constraints
        for i in 1..3 {
            res += if con.plain_dim_struck(i) { "/1" } else { "/0" };
        }
    }
    res += "\n$EndConstraints\n\n$Forces";

    let forces = elas.conc_forces();
    res += &format!("\ncount {}", forces.len());
    for (n, frc) in forces {
        res += &format!("\n{} {:.3}", n, frc[0]);

        for i in (0..dim).skip(1) {
            res += &format!("/{:.3}", frc[i]);
        }
    }
    res.push_str("\n$EndForces\n\n$DistForces");

    let dist_forces = elas.dist_forces();
    res += &format!("\ncount {}", dist_forces.len());
    for ((na, nb), frc) in dist_forces {
        res += &format!("\n{} {} {:.3}", na, nb, frc[0]);

        for i in (0..dim).skip(1) {
            res += &format!("/{:.3}", frc[i]);
        }
    }
    res += "\n$EndDistForces";

    res
}

fn name_and_num<'a>(no: usize, line: &'a str) -> Result<(&'a str, usize), FileError> {
    //convert <name x> to (name, x)
    let mut l_split = line.split(" ");
    let name = l_split.next().ok_or(parse_error(no, "no name on line"))?;
    let num_str = l_split
        .next()
        .ok_or(parse_error(no, "expected uint with label"))?;
    let num = num_str
        .parse::<usize>()
        .or(Err(parse_error(no, "bad uint with label")))?;
    Ok((name, num))
}

fn name_and_name<'a>(no: usize, line: &'a str) -> Result<(&'a str, &'a str), FileError> {
    let mut l_split = line.split(" ");
    let n = l_split.next().ok_or(parse_error(no, "no name on line"))?;
    let m = l_split
        .next()
        .ok_or(parse_error(no, "expected second name on line"))?;

    Ok((n, m))
}

fn name_and_float<'a>(no: usize, line: &'a str) -> Result<(&'a str, f64), FileError> {
    //convert <name x> to (name, x)
    let mut l_split = line.split(" ");
    let name = l_split.next().ok_or(parse_error(no, "no name on line"))?;
    let num_str = l_split
        .next()
        .ok_or(parse_error(no, "expected float with label"))?;
    let num = num_str
        .parse::<f64>()
        .or(Err(parse_error(no, "bad float with label")))?;
    Ok((name, num))
}

fn id_and_point(no: usize, line: &str) -> Result<(usize, Point), FileError> {
    // convert <id x/y/z> to (id, Point)
    let mut l_split = line.split(" ");
    let id = l_split
        .next()
        .ok_or(parse_error(no, "expected id"))?
        .parse::<usize>()
        .or(Err(parse_error(no, "bad uint")))?;

    let vals_str = l_split
        .next()
        .ok_or(parse_error(no, "expected float series"))?;
    let mut vals: Vec<f64> = Vec::new();
    for val_str in vals_str.split("/") {
        vals.push(
            val_str
                .parse::<f64>()
                .or(Err(parse_error(no, "bad float")))?,
        );
    }

    let p: Point = vals
        .try_into()
        .or(Err(parse_error(no, "couldn't assemble Point")))?;

    Ok((id, p))
}

fn id_id_and_point(no: usize, line: &str) -> Result<(usize, usize, Point), FileError> {
    // convert <id x/y/z> to (id, Point)
    let mut l_split = line.split(" ");
    let ida = l_split
        .next()
        .ok_or(parse_error(no, "expected id"))?
        .parse::<usize>()
        .or(Err(parse_error(no, "bad uint")))?;

    let idb = l_split
        .next()
        .ok_or(parse_error(no, "expected id"))?
        .parse::<usize>()
        .or(Err(parse_error(no, "bad uint")))?;

    let vals_str = l_split
        .next()
        .ok_or(parse_error(no, "expected float series"))?;
    let mut vals: Vec<f64> = Vec::new();
    for val_str in vals_str.split("/") {
        vals.push(
            val_str
                .parse::<f64>()
                .or(Err(parse_error(no, "bad float")))?,
        );
    }

    let p: Point = vals
        .try_into()
        .or(Err(parse_error(no, "couldn't assemble Point")))?;

    Ok((ida, idb, p))
}

fn id_type_and_list(no: usize, line: &str) -> Result<(usize, u8, Vec<usize>), FileError> {
    // convert <id t a/b/c...> to (id, type, Vec<nodes>) for element definition lines
    let mut l_split = line.split(" ");
    let id = l_split
        .next()
        .ok_or(parse_error(no, "expected id"))?
        .parse::<usize>()
        .or(Err(parse_error(no, "bad uint")))?;

    let tp = l_split
        .next()
        .ok_or(parse_error(no, "expected type"))?
        .parse::<u8>()
        .or(Err(parse_error(no, "bad uint")))?;

    let ns_str = l_split
        .next()
        .ok_or(parse_error(no, "expected id series"))?;
    let mut ns: Vec<usize> = Vec::new();
    for n_str in ns_str.split("/") {
        ns.push(
            n_str
                .parse::<usize>()
                .or(Err(parse_error(no, "bad uint")))?,
        );
    }

    Ok((id, tp, ns))
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

fn make_material(no: usize, mat: &str) -> Result<Material, FileError> {
    match mat {
        "AL6061" => Ok(AL6061),
        _ => Err(parse_error(no, "unsupported material name")),
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
