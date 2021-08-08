use std::collections::HashMap;

use crate::element::loading::Constraint;
use crate::element::material::Material;
use crate::mesher::bounds::{PlaneBoundary, VId};

use super::FileError;

// parser for a simple boundary definition file

pub fn lines_to_bounds<'a, T>(mut lines: T) -> Result<PlaneBoundary, FileError>
where
    T: Iterator<Item = &'a str>,
{
    let mut b = PlaneBoundary::new();
    let mut named_vertices: HashMap<String, VId> = HashMap::new();
    let mut no = 0;

    while let Some(line) = lines.next() {
        no += 1;
        if line.is_empty() || line.starts_with("#") {
            continue;
        }

        let words: Vec<&str> = line.split(" ").collect();

        match words[0] {
            "polygon" => {
                // read a sequence of lines, with a point on each line
                if words.len() != 1 {
                    return Err(FileError::parse("unexpected word after polygon"));
                }

                let mut ended = false;
                let mut poly: Vec<(f64, f64)> = Vec::new();
                let mut labels: Vec<Option<String>> = Vec::new();
                while let Some(ln) = lines.next() {
                    no += 1;

                    if ln == "end" {
                        ended = true;
                        break;
                    }
                    let wds: Vec<&str> = ln.split(" ").collect();
                    if wds.len() == 2 || wds.len() == 3 {
                        if let (Ok(a), Ok(b)) = (wds[0].parse::<f64>(), wds[1].parse::<f64>()) {
                            poly.push((a, b));

                            labels.push(if wds.len() == 2 {
                                None
                            } else {
                                Some(wds[2].to_string())
                            });
                        } else {
                            return Err(FileError::parse("could not parse point values"));
                        }
                    } else {
                        return Err(FileError::parse(format!(
                            "expected two or three words on polygon line (line {})",
                            no
                        )));
                    }
                }

                if !ended {
                    return Err(FileError::parse("ran out of lines before polygon end"));
                }

                let ids = b.store_polygon(poly);

                for (id, l) in ids.into_iter().zip(labels.into_iter()) {
                    if let Some(label) = l {
                        named_vertices.insert(label, id);
                    }
                }
            }
            "thickness" => {
                if words.len() != 2 {
                    return Err(FileError::parse("expected thickness value"));
                }
                let val: f64 = words[1]
                    .parse()
                    .or_else(|_| Err(FileError::parse("invalid thickness value")))?;
                b.set_thickness(val);
            }
            "material" => {
                if words.len() != 2 {
                    return Err(FileError::parse("expected material type"));
                }
                let mat: Material = words[1]
                    .parse()
                    .or_else(|_| Err(FileError::parse("invalid material specifier")))?;
                b.set_material(mat);
            }
            "distributed_force" => {
                if words.len() != 5 {
                    return Err(FileError::parse("expected vertex labels, force vector"));
                }
                if let (Some(ida), Some(idb)) = (
                    named_vertices.get(&words[1].to_string()).cloned(),
                    named_vertices.get(&words[2].to_string()).cloned(),
                ) {
                    if let (Ok(x), Ok(y)) = (words[3].parse::<f64>(), words[4].parse::<f64>()) {
                        b.store_distributed_force(ida, idb, (x, y).into());
                    } else {
                        return Err(FileError::parse("invalid force value"));
                    }
                } else {
                    return Err(FileError::parse("invalid vertex labels"));
                }
            }
            "distributed_constraint" => {
                if words.len() != 4 {
                    return Err(FileError::parse("expected vertex labels, constraint"));
                }
                if let (Some(ida), Some(idb)) = (
                    named_vertices.get(&words[1].to_string()).cloned(),
                    named_vertices.get(&words[2].to_string()).cloned(),
                ) {
                    if let Ok(c) = words[3].parse::<Constraint>() {
                        b.store_distributed_constraint(ida, idb, c);
                    } else {
                        return Err(FileError::parse("invalid constraint value"));
                    }
                } else {
                    return Err(FileError::parse("invalid vertex labels"));
                }
            }
            s => return Err(FileError::parse(format!("unsupported specifier: {:?}", s))),
        }
    }

    Ok(b)
}
