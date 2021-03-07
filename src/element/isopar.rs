use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::{Point, hull};
use std::convert::TryInto;
use super::material::{Material, ProblemType};
use crate::matrix::inverse::Inverse;
use super::strain::StrainRule;

// constructors for isoparametric finite element matrices

// main API - storing lots of stuff for now, can slim down with references later
// TODO replace node_points with a reference to the global node vector
#[derive(Debug, Clone)]
pub struct IsoparElement {
    node_idxs: Vec<usize>,
    node_points: Vec<Point>,
    el_type: ElementType,
    strain_rule: StrainRule,
    material: Material,
}

#[derive(Debug, Clone, Copy)]
pub enum ElementType {
    Bar2Node,
    PlaneNNode,
}

pub struct ElementMats {
    b: LinearMatrix,
    h: Option<LinearMatrix>,
    j: LinearMatrix,
    j_inv: LinearMatrix,
    det_j: f64,
}

impl IsoparElement {
    pub fn new(all_nodes: &Vec<Point>, node_idxs: Vec<usize>, material: Material) -> Self {
        // initialize an element do the logic to arrange the element type

        let node_points: Vec<Point> = node_idxs.iter()
            .map(|x| all_nodes.get(*x).expect("out of bounds node index"))
            .cloned().collect();

        let (el_type, ordering) = arrange_element(&node_points);

        // sort the indices and points with the ordering
        let node_idxs: Vec<usize> = ordering.iter().map(|x| node_idxs[*x].clone()).collect();
        let node_points: Vec<Point> = ordering.iter().map(|x| node_points[*x].clone()).collect();

        // TODO add ability to choose axisym/planestrain/planestress for 2d elements
        let strain_rule = match el_type {
            ElementType::Bar2Node => StrainRule::Bar,
            ElementType::PlaneNNode => StrainRule::PlaneStress,
        };

        IsoparElement {node_idxs, node_points, el_type, strain_rule, material}
    }

    pub fn dim(&self) -> usize {
        // uhh maybe give this its own field but storage is getting big?
        self.node_points.first().unwrap().dim()
    } 

    pub fn i_to_dof(&self, i: usize) -> (usize, usize) {
        // finds the node idx (global) and its degree of freedom for the given index (say, in K)
        (self.node_idxs[i / self.dim()], i % self.dim())
    }

    pub fn nodes(&self) -> Vec<Point> {
        self.node_points.clone()
    }

    pub fn node(&self, i: usize) -> Point {
        self.node_points[i]
    }

    pub fn node_idxs(&self) -> Vec<usize> {
        // the node indices in the global list
        self.node_idxs.clone()
    }

    pub fn node_count(&self) -> usize {
        self.node_idxs.len()
    }

    pub fn integration_order(&self) -> usize {
        // choose the gauss integration order appropriate for the element
        // simple for now
        2
    }

    pub fn h_and_grad(&self, nat_coor: Point) -> Vec<(f64, Point)> {
        // depending on underlying type, return h and its gradient
        self.el_type.h_and_grad(nat_coor, self.node_count())
    }

    pub fn find_mats(&self, nat_coor: Point) -> ElementMats {
        // initialize dim (spatial dimension count) and n (node count)
        let dim = self.dim();
        let n = self.node_count();
        let nodes = self.nodes();
        let strain_rule = self.strain_rule;

        // initialize matrices for construction
        let mut j = LinearMatrix::zeros((dim, dim));
        let mut nat_grad_interps = Vec::new();
        for i in 0..dim {
            nat_grad_interps.push(LinearMatrix::zeros((dim, dim * n)));
        }
        let mut h_mat = LinearMatrix::zeros((dim, dim * n));

        let funcs = self.h_and_grad(nat_coor);
        for (i, (h, nat_grad_h)) in funcs.into_iter().enumerate() {
            let node_loc = self.node(i);
            for x in 0..dim {
                // h_mat is a simple arrangement of h values
                h_mat[(x, (dim * i) + x)] = h;
                // partial derivative of h in r/s/t (x is current dimension)
                let nat_grad_in_dim = nat_grad_h[x];

                for y in 0..dim {
                    // J[x, y] is d (x, y, z)[y] / d (r, s, t)[x]
                    // values interpolated from (d h_i / d (r/s/t)) * (x/y/z)_i
                    j[(x, y)] += nat_grad_in_dim * node_loc[y];

                    // fill out natural (r/s/t) gradient interpolations
                    // by the end, ngi[y] * U = grad_rst (u, v, w)[y]
                    nat_grad_interps[y][(x, (dim * i) + y)] = nat_grad_in_dim;
                }
            }
        }

        // invert J to find the gradient interpolation for the global coordinate system
        let j_inv = j.inverse();
        let det_j = j.determinant();

        // construct the strain interpolation matrix
        let mut b = LinearMatrix::zeros((strain_rule.vec_len(), dim * n));

        for (i, nat_grad_interp) in nat_grad_interps.into_iter().enumerate() {
            // ngi is the natural coordinate gradient interpolation for (u, v, w)[i]
            let grad_interp: LinearMatrix = j_inv.mul(&nat_grad_interp);
            //println!("grad_interp:\n{}", grad_interp);
            // each row of gi is an interpolation for d(u/v/w)/dx, d(u/v/w)/dy ...
            for j in 0..dim {
                // if ngi[j, ..] should be added to a row in b, add it
                if let Some(idx) = strain_rule.dest_idx(i, j) {
                    for (k, coeff) in grad_interp.row(j).cloned().enumerate() {
                        //println!("adding to b[({}, {})]:\n{}", idx, k, coeff);
                        b[(idx, k)] += coeff;
                    }
                }
            }
        }
        // return it all :)
        ElementMats {b, h: Some(h_mat), j, j_inv, det_j}
    }

    pub fn find_k_integrand(&self, nat_coor: Point) -> LinearMatrix {
        let mut mats = self.find_mats(nat_coor);

        let material_problem = match self.strain_rule {
            StrainRule::Bar => ProblemType::Bar,
            StrainRule::PlaneStress => ProblemType::PlaneStress,
            StrainRule::PlaneStrain => ProblemType::PlaneStrain,
            StrainRule::ThreeDimensional => ProblemType::ThreeDimensional,
        };

        // TODO compute C once and store for performance improvement
        let c = self.material.get_c(material_problem);

        // the integrand for K is (det J) * B_t * C * B
        mats.b.transpose(); // transpose in place is cheap
        let mut inter: LinearMatrix = mats.b.mul(&c);
        inter *= mats.det_j;
        mats.b.transpose();

        inter.mul(&mats.b)
    }
}

fn arrange_element(nodes: &Vec<Point>) -> (ElementType, Vec<usize>) {
    // find the appropriate element type to use for the node arrangement and the canonical node ordering
    if let Some(p) = nodes.first() {
        match p.dim() {
            1 => arrange_element_1d(nodes),
            2 => arrange_element_2d(nodes),
            3 => unimplemented!("3D elements unavailable"),
            _ => panic!("unreachable"),
        }
    }
    else {
        panic!("empty node list")
    }
}

fn arrange_element_1d(nodes: &Vec<Point>) -> (ElementType, Vec<usize>) {
    match nodes.len() {
        2 => (ElementType::Bar2Node, vec![0, 1]),
        _ => unimplemented!("N-node bar unavailable"),
    }
}

fn arrange_element_2d(nodes: &Vec<Point>) -> (ElementType, Vec<usize>) {
    // TODO handle higher-order elements
    match nodes.len() {
        4 => {
            let node_hull = hull::jarvis_hull(nodes);
            if node_hull.len() == 4 {
                // all nodes are on the hull
                (ElementType::PlaneNNode, node_hull)
            }
            else {
                panic!("nonconvex element")
            }
        }
        _ => unimplemented!("N-node plane element unavailable")
    }
}

impl ElementType {
    fn h_and_grad(&self, nat_coor: Point, node_count: usize) -> Vec<(f64, Point)> {
        match self {
            ElementType::Bar2Node => {
                vec![
                    (0.5 * (1.0 - nat_coor[0]), Point::new(&[-0.5])),
                    (0.5 * (1.0 + nat_coor[0]), Point::new(&[0.5]))
                ]
            },
            ElementType::PlaneNNode => {
                plane_n_node_h_and_grad(nat_coor, node_count)
            }
        }
    }
}

impl ElementMats {
    pub fn det_j(&self) -> f64 {
        self.det_j
    }
}

// TODO there has to be a better way
fn plane_n_node_h_and_grad(nat_coor: Point, node_count: usize) -> Vec<(f64, Point)> {
    // calculate interpolation functions and their derivatives
    // interpolation is x = sum( h_i(r, s)*x_i )
    // returns (h_i, (dh_i/dr, dh_i/ds)) for i in 0..n
    if node_count < 4 { panic!("bad node count") }

    let (r, s) = nat_coor.try_into().unwrap();

    let mut h = vec![0.0; node_count];
    let mut dhdr = vec![0.0; node_count];
    let mut dhds = vec![0.0; node_count];

    h[0] = 0.25 * (1.0 + r) * (1.0 + s);
    dhdr[0] = 0.25 * (1.0 + s);
    dhds[0] = 0.25 * (1.0 + r);

    h[1] = 0.25 * (1.0 - r) * (1.0 + s);
    dhdr[1] = -0.25 * (1.0 + s);
    dhds[1] = 0.25 * (1.0 - r);

    h[2] = 0.25 * (1.0 - r) * (1.0 - s);
    dhdr[2] = -0.25 * (1.0 - s);
    dhds[2] = -0.25 * (1.0 - r);

    h[3] = 0.25 * (1.0 + r) * (1.0 - s);
    dhdr[3] = 0.25 * (1.0 - s);
    dhds[3] = -0.25 * (1.0 + r);

    if node_count <= 4 {
        return collect_pnn_hag(h, dhdr, dhds)
    }

    let temp_h = 0.5 * (1.0 - (r * r)) * (1.0 + s);
    let temp_dhdr = 0.5 * (-2.0 * r * (1.0 + s));
    let temp_dhds = 0.5 * (1.0 - (r * r));
    h[4] = temp_h;
    dhdr[4] = temp_dhdr;
    dhds[4] = temp_dhds;
    h[0] -= 0.5 * temp_h;
    dhdr[0] -= 0.5 * temp_h;
    dhds[0] -= 0.5 * temp_h;
    h[1] -= 0.5 * temp_h;
    dhdr[1] -= 0.5 * temp_h;
    dhds[1] -= 0.5 * temp_h;

    if node_count <= 5 {
        return collect_pnn_hag(h, dhdr, dhds)
    }

    let temp_h = 0.5 * (1.0 - (s * s)) * (1.0 - r);
    let temp_dhdr = -0.5 * (1.0 - (s * s));
    let temp_dhds = 0.5 * (-2.0 * s * (1.0 - r));
    h[5] = temp_h;
    dhdr[5] = temp_dhdr;
    dhds[5] = temp_dhds;
    h[1] -= 0.5 * temp_h;
    dhdr[1] -= 0.5 * temp_h;
    dhds[1] -= 0.5 * temp_h;
    h[2] -= 0.5 * temp_h;
    dhdr[2] -= 0.5 * temp_h;
    dhds[2] -= 0.5 * temp_h;

    if node_count <= 6 {
        return collect_pnn_hag(h, dhdr, dhds)
    }

    let temp_h = 0.5 * (1.0 - (r * r)) * (1.0 - s);
    let temp_dhdr = 0.5 * (-2.0 * r * (1.0 - s));
    let temp_dhds = -0.5 * (1.0 - (r * r));
    h[6] = temp_h;
    dhdr[6] = temp_dhdr;
    dhds[6] = temp_dhds;
    h[2] -= 0.5 * temp_h;
    dhdr[2] -= 0.5 * temp_h;
    dhds[2] -= 0.5 * temp_h;
    h[3] -= 0.5 * temp_h;
    dhdr[3] -= 0.5 * temp_h;
    dhds[3] -= 0.5 * temp_h;

    if node_count <= 7 {
        return collect_pnn_hag(h, dhdr, dhds)
    }

    let temp_h = 0.5 * (1.0 - (s * s)) * (1.0 + r);
    let temp_dhdr = 0.5 * (1.0 - (s * s));
    let temp_dhds = 0.5 * (-2.0 * s * (1.0 + r));
    h[7] = temp_h;
    dhdr[7] = temp_dhdr;
    dhds[7] = temp_dhds;
    h[0] -= 0.5 * temp_h;
    dhdr[0] -= 0.5 * temp_h;
    dhds[0] -= 0.5 * temp_h;
    h[3] -= 0.5 * temp_h;
    dhdr[3] -= 0.5 * temp_h;
    dhds[3] -= 0.5 * temp_h;

    if node_count <= 7 {
        return collect_pnn_hag(h, dhdr, dhds)
    }

    let temp_h = (1.0 - (r * r)) * (1.0 - (s * s));
    let temp_dhdr = -2.0 * r * (1.0 - (s * s));
    let temp_dhds = -2.0 * s * (1.0 - (r * r));
    h[8] = temp_h;
    dhdr[8] = temp_dhdr;
    dhds[8] = temp_dhds;
    for i in 0..4 {
        h[i] -= 0.25 * temp_h;
        dhdr[i] -= 0.25 * temp_dhdr;
        dhds[i] -= 0.25 * temp_dhds;
    }
    for i in 4..8 {
        h[i] -= 0.5 * temp_h;
        dhdr[i] -= 0.5 * temp_dhdr;
        dhds[i] -= 0.5 * temp_dhds;
    }

    return collect_pnn_hag(h, dhdr, dhds)
}

fn collect_pnn_hag(h: Vec<f64>, dhdr: Vec<f64>, dhds: Vec<f64>) -> Vec<(f64, Point)> {
    h.into_iter()
        .zip(dhdr.into_iter().zip(dhds.into_iter()))
        .map(|(x, y)| (x, y.into()))
        .collect()
}