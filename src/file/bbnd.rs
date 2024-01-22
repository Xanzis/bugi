use std::collections::HashMap;

use crate::element::constraint::Constraint;
use crate::element::material::Material;
use crate::element::strain::Condition;
use crate::mesher::bounds::PlaneBoundary;
use crate::mesher::Vertex;

use super::FileError;

use nom::{
    self,
    branch::alt,
    bytes::complete::tag,
    character::complete::{self, alphanumeric1},
    combinator::map,
    multi::{separated_list0, separated_list1},
    number::complete::double,
    sequence::{delimited, preceded, separated_pair, tuple},
};

// parser for a simple boundary definition file

#[derive(Clone, Debug)]
enum ParseItem<'a> {
    Polygon(Vec<ParseVertex<'a>>),
    Condition(Condition),
    Material(Material),
    DistForce(&'a str, &'a str, f64, f64),
    DistConstraint(&'a str, &'a str, Constraint),
}

impl<'a> ParseItem<'a> {
    fn polygon(&'a self) -> Option<&'a [ParseVertex]> {
        match *self {
            ParseItem::Polygon(ref x) => Some(x),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ParseVertex<'a> {
    Plain(f64, f64),
    Labeled(f64, f64, &'a str),
}

impl<'a> ParseVertex<'a> {
    fn with_label(&self) -> (f64, f64, Option<&'a str>) {
        match *self {
            ParseVertex::Plain(a, b) => (a, b, None),
            ParseVertex::Labeled(a, b, c) => (a, b, Some(c)),
        }
    }
}

pub fn bbnd_to_bounds(file: &str) -> Result<PlaneBoundary, FileError> {
    let items = parse_file(file)?;
    let mut label_map: HashMap<&str, Vertex> = HashMap::new();
    let mut res = PlaneBoundary::new();

    // two passes, first one gets nodes and gets labels / vids
    let mut vertex_points: Vec<(f64, f64)> = Vec::new();
    let mut vertex_labels: Vec<Option<&str>> = Vec::new();
    for item in items.iter() {
        if let Some(vs) = item.polygon() {
            vertex_points.clear();
            vertex_labels.clear();
            for v in vs {
                let (x, y, l) = v.with_label();
                vertex_points.push((x, y));
                vertex_labels.push(l);
            }

            let vertices = res.store_polygon(&vertex_points);
            for (&l, vertex) in vertex_labels.iter().zip(vertices) {
                if let Some(label) = l {
                    label_map.insert(label, vertex);
                }
            }
        }
    }

    // second pass to fill in the details
    for item in items.iter() {
        match *item {
            ParseItem::Polygon(_) => {}
            ParseItem::Condition(x) => {
                res.set_condition(x);
            }
            ParseItem::Material(x) => {
                res.set_material(x);
            }
            ParseItem::DistForce(a, b, x, y) => {
                let a_vertex = label_map
                    .get(&a)
                    .ok_or_else(|| FileError::parse("invalid vertex label"))?;
                let b_vertex = label_map
                    .get(&b)
                    .ok_or_else(|| FileError::parse("invalid vertex label"))?;
                res.store_distributed_force(*a_vertex, *b_vertex, (x, y).into());
            }
            ParseItem::DistConstraint(a, b, c) => {
                let a_vertex = label_map
                    .get(&a)
                    .ok_or_else(|| FileError::parse("invalid vertex label"))?;
                let b_vertex = label_map
                    .get(&b)
                    .ok_or_else(|| FileError::parse("invalid vertex label"))?;
                res.store_distributed_constraint(*a_vertex, *b_vertex, c);
            }
        }
    }

    Ok(res)
}

fn parse_file<'a>(file: &'a str) -> Result<Vec<ParseItem<'a>>, FileError> {
    let condition = alt((
        map(preceded(tag("condition planestrain "), double), |t: f64| {
            ParseItem::Condition(Condition::PlaneStrain(t))
        }),
        map(preceded(tag("condition planestress "), double), |t: f64| {
            ParseItem::Condition(Condition::PlaneStress(t))
        }),
        map(tag("condition axisymmetric"), |_| {
            ParseItem::Condition(Condition::Axisymmetric)
        }),
    ));
    // TODO fix material error handling
    let material = map(preceded(tag("material "), alphanumeric1), |m: &str| {
        ParseItem::Material(m.parse::<Material>().unwrap())
    });

    let dist_force = map(
        preceded(
            tag("distributed_force "),
            tuple((
                alphanumeric1,
                preceded(complete::char(' '), alphanumeric1),
                preceded(complete::char(' '), double),
                preceded(complete::char(' '), double),
            )),
        ),
        |f: (&str, &str, f64, f64)| ParseItem::DistForce(f.0, f.1, f.2, f.3),
    );

    let dist_constraint = map(
        preceded(
            tag("distributed_constraint "),
            tuple((
                alphanumeric1,
                preceded(complete::char(' '), alphanumeric1),
                preceded(complete::char(' '), alphanumeric1),
            )),
        ),
        |f: (&str, &str, &str)| {
            ParseItem::DistConstraint(f.0, f.1, f.2.parse::<Constraint>().unwrap())
        },
    );

    let vertex = alt((
        map(
            tuple((
                double,
                preceded(complete::char(' '), double),
                preceded(complete::char(' '), alphanumeric1),
            )),
            |v: (f64, f64, &str)| ParseVertex::Labeled(v.0, v.1, v.2),
        ),
        map(
            separated_pair(double, complete::char(' '), double),
            |v: (f64, f64)| ParseVertex::Plain(v.0, v.1),
        ),
    ));

    let polygon = delimited(
        tag("polygon\n"),
        map(separated_list1(complete::char('\n'), vertex), |v_list| {
            ParseItem::Polygon(v_list)
        }),
        tag("\nend"),
    );

    let mut parser = separated_list0(
        tag("\n"),
        alt((polygon, condition, material, dist_force, dist_constraint)),
    );
    let (rem, parsed) = parser(file)
        .map_err(|_e: nom::Err<nom::error::Error<_>>| FileError::parse("parse error"))?;
    if !rem.is_empty() {
        Err(FileError::parse("parser did not consume entire file\n{}"))
    } else {
        Ok(parsed)
    }
}
