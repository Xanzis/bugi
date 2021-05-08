use super::integrate::{self, NdGaussSamples};
use super::material::{Material, ProblemType};
use super::strain::StrainRule;
use super::stress::StressState;
use crate::matrix::inverse::Inverse;
use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::{hull, predicates, Point};
use std::convert::TryInto;

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
    Triangle3Node,
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

        let node_points: Vec<Point> = node_idxs
            .iter()
            .map(|x| all_nodes.get(*x).expect("out of bounds node index"))
            .cloned()
            .collect();

        let (el_type, ordering) = arrange_element(&node_points);

        // sort the indices and points with the ordering
        let node_idxs: Vec<usize> = ordering.iter().map(|x| node_idxs[*x].clone()).collect();
        let node_points: Vec<Point> = ordering.iter().map(|x| node_points[*x].clone()).collect();

        // TODO add ability to choose axisym/planestrain/planestress for 2d elements
        let strain_rule = match el_type {
            ElementType::Bar2Node => StrainRule::Bar,
            ElementType::PlaneNNode => StrainRule::PlaneStress,
            ElementType::Triangle3Node => StrainRule::PlaneStress,
        };

        IsoparElement {
            node_idxs,
            node_points,
            el_type,
            strain_rule,
            material,
        }
    }

    pub fn dim(&self) -> usize {
        // uhh maybe give this its own field but storage is getting big?
        self.node_points.first().unwrap().dim()
    }

    pub fn dofs(&self) -> usize {
        // return the number of degrees of freedom in the element
        self.dim() * self.node_idxs.len()
    }

    pub fn i_to_dof(&self, i: usize) -> (usize, usize) {
        // finds the node idx (global) and its degree of freedom for the given index (say, in K)
        if i >= self.dofs() {
            panic!("out of bounds dof request");
        }
        (self.node_idxs[i / self.dim()], i % self.dim())
    }

    pub fn nodes(&self) -> Vec<Point> {
        self.node_points.clone()
    }

    pub fn node(&self, i: usize) -> Point {
        self.node_points[i]
    }

    fn node_natural(&self, i: usize) -> Point {
        // returns the node's natural coordinates
        match self.el_type {
            ElementType::Bar2Node => match i {
                0 => Point::new(&[-1.0]),
                1 => Point::new(&[1.0]),
                _ => panic!("out of range node request"),
            },
            ElementType::Triangle3Node => match i {
                0 => Point::new(&[0.0, 0.0]),
                1 => Point::new(&[1.0, 0.0]),
                2 => Point::new(&[0.0, 1.0]),
                _ => panic!("out of range node request"),
            },
            ElementType::PlaneNNode => match i {
                0 => Point::new(&[1.0, 1.0]),
                1 => Point::new(&[-1.0, 1.0]),
                2 => Point::new(&[-1.0, -1.0]),
                3 => Point::new(&[1.0, -1.0]),
                4 => Point::new(&[0.0, 1.0]),
                5 => Point::new(&[-1.0, 0.0]),
                6 => Point::new(&[0.0, -1.0]),
                7 => Point::new(&[1.0, 0.0]),
                8 => Point::new(&[0.0, 0.0]),
                _ => panic!("out of range node request"),
            },
        }
    }

    pub fn nodes_connect(&self, i: usize, j: usize) -> bool {
        // tests if two nodes are connected by an edge
        // TODO this is a slow naive implentation
        self.edges().unwrap().into_iter().any(|x| x == (i, j) || x == (j, i))
    }

    pub fn node_idxs(&self) -> Vec<usize> {
        // the node indices in the global list
        self.node_idxs.clone()
    }

    pub fn node_idx(&self, i: usize) -> usize {
        self.node_idxs[i]
    }

    pub fn node_count(&self) -> usize {
        self.node_idxs.len()
    }

    fn global_to_own_idx(&self, i: usize) -> Option<usize> {
        // convert a node index in the global list to an index in the local one
        // TODO this is slow
        self.node_idxs.iter().position(|&x| x == i)
    }

    pub fn integration_order(&self) -> usize {
        // choose the gauss integration order appropriate for the element
        // simple for now
        // TODO be more clever about this
        2
    }

    pub fn edges(&self) -> Option<Vec<(usize, usize)>> {
        // pairs of indices in the global point list which give the element edges
        match self.el_type {
            ElementType::Bar2Node => Some(vec![(self.node_idxs[0], self.node_idxs[1])]),
            ElementType::PlaneNNode => {
                let mut res = Vec::new();
                for i in 0..(self.node_idxs.len() - 1) {
                    res.push((self.node_idxs[i], self.node_idxs[i + 1]));
                }
                res.push((self.node_idxs[0], self.node_idxs.last().unwrap().clone()));
                Some(res)
            }
            ElementType::Triangle3Node => Some(vec![
                (self.node_idxs[0], self.node_idxs[1]),
                (self.node_idxs[1], self.node_idxs[2]),
                (self.node_idxs[2], self.node_idxs[0]),
            ]),
        }
    }

    pub fn triangles(&self) -> Option<Vec<(usize, usize, usize)>> {
        // triplets of indices in the global point list which give the element face triangles
        match self.el_type {
            ElementType::Bar2Node => None,
            ElementType::PlaneNNode => plane_n_node_triangles(&self.node_idxs),
            ElementType::Triangle3Node => Some(vec![(
                self.node_idxs[0],
                self.node_idxs[1],
                self.node_idxs[2],
            )]),
        }
    }

    fn h_and_grad(&self, nat_coor: Point) -> Vec<(f64, Point)> {
        // depending on underlying type, return h and its gradient
        self.el_type.h_and_grad(nat_coor, self.node_count())
    }

    pub fn find_mats(&self, nat_coor: Point) -> ElementMats {
        // initialize dim (spatial dimension count) and n (node count)
        let dim = self.dim();
        let n = self.node_count();
        let strain_rule = self.strain_rule;

        // initialize matrices for construction
        let mut j = LinearMatrix::zeros((dim, dim));
        let mut nat_grad_interps = Vec::new();
        for _ in 0..dim {
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
        let det_j = j
            .determinant()
            .expect("unreachable - element determinants should be nonzero");

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
        ElementMats {
            b,
            h: Some(h_mat),
            j,
            j_inv,
            det_j,
        }
    }

    pub fn jacobian_ratio(&self) -> f64 {
        // returns the ratio of smallest jacobian to largest
        // with an edge case for opposite-sign jacobians
        let samples = NdGaussSamples::new(self.dim(), self.integration_order());
        let jacobians: Vec<f64> = samples.map(|(p, _)| self.find_mats(p).det_j).collect();

        // unwrap here is ok because jacobians should never be an empty list
        let a = jacobians
            .iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        let b = jacobians
            .iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();

        // note this may return a negative number
        // if so, the element is likely unusably distorted, handle this downstream
        if a.abs() < b.abs() {
            a / b
        } else {
            b / a
        }
    }

    // TODO this should be an API somewhere else
    // ProblemType / StrainRule / StressType are all doing similar things and should be combined
    fn material_problem(&self) -> ProblemType {
        match self.strain_rule {
            StrainRule::Bar => ProblemType::Bar,
            StrainRule::PlaneStress => ProblemType::PlaneStress,
            StrainRule::PlaneStrain => ProblemType::PlaneStrain,
            StrainRule::ThreeDimensional => ProblemType::ThreeDimensional,
        }
    }

    pub fn find_k_integrand(&self, nat_coor: Point) -> LinearMatrix {
        let mut mats = self.find_mats(nat_coor);

        // TODO compute C once and store for performance improvement
        let c = self.material.get_c(self.material_problem());

        // the integrand for K is (det J) * B_t * C * B
        mats.b.transpose(); // transpose in place is cheap
        let mut inter: LinearMatrix = mats.b.mul(&c);
        inter *= mats.det_j;
        mats.b.transpose();

        inter.mul(&mats.b)
    }

    pub fn find_f_l(&self, a_idx: usize, b_idx: usize, f: &LinearMatrix) -> LinearMatrix {
        // integrate the interpolation of a distributed force over an element edge
        // a_idx and b_idx are node indices of the edge ends in the global node array

        if !self.nodes_connect(a_idx, b_idx) {
            panic!("invalid edge");
        }

        if f.shape() != (self.dim(), 1) {
            panic!("dsitributed force must have dimensions ({}, 1)", self.dim());
        }

        // find the node points, first converting to the local point indices
        let a = self.node_natural(self.global_to_own_idx(a_idx).expect("element does not contain node"));
        let b = self.node_natural(self.global_to_own_idx(b_idx).expect("element does not contain node"));

        // MISTAKE: find_mats takes natural coordinates
        let integrand = |p: Point| {
            let mats = self.find_mats(p);
            let mut h = mats.h.unwrap();
            h.transpose();
            h.mul(f)
        };

        integrate::gauss_segment_mat(integrand, a, b, self.integration_order())
    }

    pub fn node_stress(&self, node_idx: usize, u: Vec<f64>) -> StressState {
        // find the stress of the given node (node_idx is the index in the local node list)
        // u is a displacement vector like [u1 v1 w1 u2 v2 w2 ...]
        // TODO better interface for constructing U, elas handles it right now

        let nat_coor = self.node_natural(node_idx);
        let mats = self.find_mats(nat_coor);

        let u = LinearMatrix::from_flat((u.len(), 1), u);
        let c = self.material.get_c(self.material_problem());

        // compute CBU
        let mut res = c;
        res = res.mul(&mats.b);
        res = res.mul(&u);
        StressState::from_vector(res)
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
    } else {
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
        3 => {
            let (p, q, r) = (nodes[0], nodes[1], nodes[2]);
            match predicates::triangle_dir((p, q, r)) {
                predicates::Orient::Positive => (ElementType::Triangle3Node, vec![0, 1, 2]),
                predicates::Orient::Negative => (ElementType::Triangle3Node, vec![2, 1, 0]),
                predicates::Orient::Zero => panic!("degenerate element"),
            }
        }
        4 => {
            let node_hull = hull::jarvis_hull(nodes);
            if node_hull.len() == 4 {
                // all nodes are on the hull
                (ElementType::PlaneNNode, node_hull)
            } else {
                panic!("nonconvex element")
            }
        }
        _ => unimplemented!("N-node plane element unavailable"),
    }
}

impl ElementType {
    fn h_and_grad(&self, nat_coor: Point, node_count: usize) -> Vec<(f64, Point)> {
        match self {
            ElementType::Bar2Node => vec![
                (0.5 * (1.0 - nat_coor[0]), Point::new(&[-0.5])),
                (0.5 * (1.0 + nat_coor[0]), Point::new(&[0.5])),
            ],
            ElementType::PlaneNNode => plane_n_node_h_and_grad(nat_coor, node_count),
            ElementType::Triangle3Node => triangle_3_node_h_and_grad(nat_coor),
        }
    }
}

impl ElementMats {
    pub fn det_j(&self) -> f64 {
        self.det_j
    }

    pub fn j<'a>(&'a self) -> &'a LinearMatrix {
        &self.j
    }

    pub fn j_inv<'a>(&'a self) -> &'a LinearMatrix {
        &self.j_inv
    }

    pub fn get_h<'a>(&'a self) -> Option<&'a LinearMatrix> {
        self.h.as_ref()
    }
}

// TODO there has to be a better way
fn plane_n_node_h_and_grad(nat_coor: Point, node_count: usize) -> Vec<(f64, Point)> {
    // calculate interpolation functions and their derivatives
    // interpolation is x = sum( h_i(r, s)*x_i )
    // returns (h_i, (dh_i/dr, dh_i/ds)) for i in 0..n
    if node_count < 4 {
        panic!("bad node count")
    }

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
        return collect_pnn_hag(h, dhdr, dhds);
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
        return collect_pnn_hag(h, dhdr, dhds);
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
        return collect_pnn_hag(h, dhdr, dhds);
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
        return collect_pnn_hag(h, dhdr, dhds);
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
        return collect_pnn_hag(h, dhdr, dhds);
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

    return collect_pnn_hag(h, dhdr, dhds);
}

fn collect_pnn_hag(h: Vec<f64>, dhdr: Vec<f64>, dhds: Vec<f64>) -> Vec<(f64, Point)> {
    h.into_iter()
        .zip(dhdr.into_iter().zip(dhds.into_iter()))
        .map(|(x, y)| (x, y.into()))
        .collect()
}

fn triangle_3_node_h_and_grad(nat_coor: Point) -> Vec<(f64, Point)> {
    let (r, s) = nat_coor.try_into().expect("bad point input");

    let h = [1.0 - r - s, r, s];
    let dhdr = [-1.0, 1.0, 0.0];
    let dhds = [-1.0, 0.0, 1.0];

    vec![
        (h[0], (dhdr[0], dhds[0]).into()),
        (h[1], (dhdr[1], dhds[1]).into()),
        (h[2], (dhdr[2], dhds[2]).into()),
    ]
}

fn plane_n_node_triangles(node_idxs: &Vec<usize>) -> Option<Vec<(usize, usize, usize)>> {
    // find the triplets of node indices representing regions of the n-node plane element

    match node_idxs.len() {
        4 => Some(vec![
            (node_idxs[0], node_idxs[1], node_idxs[2]),
            (node_idxs[0], node_idxs[2], node_idxs[3]),
        ]),
        _ => unimplemented!("triangles not implemented for this order of plane node"),
    }
}
