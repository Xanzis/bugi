use super::integrate::{self, NdGaussSamples};
use super::material::{Material, ProblemType};
use super::strain::StrainRule;
use super::stress::StressState;
use super::{ElementAssemblage, NodeDof, NodeId};

use crate::matrix::inverse::Inverse;
use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;
use std::convert::TryInto;

// isoparametric finite elements

#[derive(Clone, Copy, Debug)]
struct NaturalCoor(Point);

// an isoparametric triangular three-node element
// after some consideration, the only kind of element available in bugi

const EL_DIMS: usize = 2;
const EL_NODES: usize = 3;
const EL_DOFS: usize = EL_DIMS * EL_NODES;

#[derive(Debug, Clone)]
pub struct Element {
    // formerly IsoparTriangle3
    node_ids: [NodeId; EL_NODES],
    node_points: [Point; EL_NODES],
    material: Material,
    thickness: f64,
}

impl Element {
    const INT_ORDER: usize = 2;
    const STRAIN_RULE: StrainRule = StrainRule::PlaneStress;

    pub fn new(elas: &ElementAssemblage, node_ids: [NodeId; EL_NODES]) -> Self {
        let node_points = [
            elas.node(node_ids[0]),
            elas.node(node_ids[1]),
            elas.node(node_ids[2]),
        ];
        let material = elas.material();
        let thickness = elas
            .thickness()
            .expect("element assemblage has no thickness");
        Self {
            node_ids,
            node_points,
            material,
            thickness,
        }
    }

    // former impl of Isopar

    pub fn material(&self) -> Material {
        self.material
    }

    pub fn material_problem(&self) -> ProblemType {
        // TODO: may want to make plane stress / plane strain selectable
        ProblemType::PlaneStress
    }

    pub fn node_points(&self) -> [Point; EL_NODES] {
        self.node_points
    }

    pub fn node_ids(&self) -> [NodeId; EL_NODES] {
        self.node_ids
    }

    pub fn nodes_connect(&self, a: NodeId, b: NodeId) -> bool {
        let (x, y, z) = (self.node_ids[0], self.node_ids[1], self.node_ids[2]);

        [(x, y), (y, x), (y, z), (z, y), (x, z), (z, x)]
            .iter()
            .any(|&ns| ns == (a, b))
    }

    pub fn node_natural(&self, a: NodeId) -> Option<Point> {
        let (x, y, z) = (self.node_ids[0], self.node_ids[1], self.node_ids[2]);

        if a == x {
            Some((0.0, 0.0).into())
        } else if a == y {
            Some((1.0, 0.0).into())
        } else if a == z {
            Some((0.0, 1.0).into())
        } else {
            None
        }
    }

    pub fn node(&self, a: NodeId) -> Option<Point> {
        let (x, y, z) = (self.node_ids[0], self.node_ids[1], self.node_ids[2]);

        if a == x {
            Some(self.node_points[0])
        } else if a == y {
            Some(self.node_points[1])
        } else if a == z {
            Some(self.node_points[2])
        } else {
            None
        }
    }

    pub fn map_dof(&self, i: usize) -> NodeDof {
        // map an index in the canonical element dof ordering to
        // a global node and its degree of freedom (x/y)

        let node_idx = i / 2;
        let node_dof = i % 2;

        NodeDof(self.node_ids[node_idx], node_dof)
    }

    fn h_and_grad(&self, c: NaturalCoor) -> [(f64, Point); EL_NODES] {
        let (r, s) = c.0.try_into().expect("bad point input");

        let h = [1.0 - r - s, r, s];
        let dhdr = [-1.0, 1.0, 0.0];
        let dhds = [-1.0, 0.0, 1.0];

        [
            (h[0], (dhdr[0], dhds[0]).into()),
            (h[1], (dhdr[1], dhds[1]).into()),
            (h[2], (dhdr[2], dhds[2]).into()),
        ]
    }

    pub fn edges(&self) -> Vec<(NodeId, NodeId)> {
        vec![
            (self.node_ids[0], self.node_ids[1]),
            (self.node_ids[1], self.node_ids[2]),
            (self.node_ids[2], self.node_ids[0]),
        ]
    }

    pub fn triangles(&self) -> Vec<(NodeId, NodeId, NodeId)> {
        vec![(self.node_ids[0], self.node_ids[1], self.node_ids[2])]
    }

    // former default impls in Isopar

    pub fn map_matrix<T: MatrixLike>(&self, mat: &T) -> Vec<(NodeDof, NodeDof, f64)> {
        mat.indices()
            .map(|i| (self.map_dof(i.0), self.map_dof(i.1), mat[i]))
            .collect()
    }

    pub fn map_vec(&self, v: &[f64]) -> Vec<(NodeDof, f64)> {
        v.iter()
            .enumerate()
            .map(|(i, &val)| (self.map_dof(i), val))
            .collect()
    }

    // former impl of Element for IsoparElement

    pub fn calc_k(&self) -> Vec<(NodeDof, NodeDof, f64)> {
        let c = self.material().get_c(self.material_problem());

        let integrand = |nat_coor| {
            let h_and_grad = self.h_and_grad(NaturalCoor(nat_coor));
            let mut mats = find_isopar_mats(h_and_grad, self.node_points(), &Self::STRAIN_RULE);

            // the integrand for K is scaling * (det J) * B_t * C * B
            mats.b.transpose(); // transpose in place is cheap
            let mut inter: LinearMatrix = mats.b.mul(&c);
            inter *= mats.det_j;
            mats.b.transpose();
            mats.b *= self.thickness;

            inter.mul(&mats.b)
        };

        let el_k = super::integrate::nd_gauss_mat(integrand, EL_DIMS, Self::INT_ORDER);

        // process the result for handoff to the element assemblage
        self.map_matrix(&el_k)
    }

    pub fn calc_m(&self) -> Vec<(NodeDof, NodeDof, f64)> {
        let integrand = |nat_coor| {
            let h_and_grad = self.h_and_grad(NaturalCoor(nat_coor));
            let mats = find_isopar_mats(h_and_grad, self.node_points(), &Self::STRAIN_RULE);

            // integrand is rho * scaling * H_t * H
            let h = mats.h.unwrap();
            let mut ht = h.clone();
            ht.transpose();
            ht *= self.material().density() * self.thickness;

            ht.mul(&h)
        };

        let el_m = super::integrate::nd_gauss_mat(integrand, EL_DIMS, Self::INT_ORDER);

        // process the result for handoff to the element assemblage
        self.map_matrix(&el_m)
    }

    pub fn int_f_l(&self, edge: (NodeId, NodeId), f: Point) -> Option<Vec<(NodeDof, f64)>> {
        // integrate the interpolation of a distributed force over an element edge
        // a_idx and b_idx are node indices of the edge ends in the global node array

        if !self.nodes_connect(edge.0, edge.1) {
            // invalid edge
            return None;
        }

        assert_eq!(
            f.dim(),
            EL_DIMS,
            "distributed force must have {} dimensions",
            EL_DIMS
        );

        // convert f to column vector
        let f = LinearMatrix::col_vec(f.to_vec());

        // find the node points, first converting to the local point indices
        let a = self.node_natural(edge.0).unwrap(); // edge nodes already validated
        let b = self.node_natural(edge.1).unwrap();

        let integrand = |p: Point| {
            let h_and_grad = self.h_and_grad(NaturalCoor(p));
            let mats = find_isopar_mats(h_and_grad, self.node_points(), &Self::STRAIN_RULE);
            let mut h = mats.h.unwrap();
            h.transpose();
            h.mul(&f)
        };

        let mut res = integrate::gauss_segment_mat(integrand, a, b, Self::INT_ORDER);

        // rescale the integral from natural coordinates
        // divide by natural segment length, multiply by real length

        let nat_len = a.dist(b);

        let a_real = self.node(edge.0).unwrap();
        let b_real = self.node(edge.1).unwrap();
        let real_len = a_real.dist(b_real);

        res *= real_len / nat_len;
        // TODO dumb allocation
        let res: Vec<_> = res.flat().cloned().collect();
        Some(self.map_vec(&res))
    }

    pub fn stresses<'a>(
        &'a self,
        disps: Box<dyn Fn(NodeId) -> Point + 'a>,
    ) -> Vec<(NodeId, StressState)> {
        let mut res = Vec::new();

        // construct the displacement vector
        let mut u = Vec::new();
        for nid in self.node_ids() {
            let node_disp = disps(nid);
            for i in 0..EL_DIMS {
                u.push(node_disp[i]);
            }
        }

        let u = LinearMatrix::col_vec(u);

        // compute a stress at each node
        for &nid in self.node_ids().iter() {
            let nat_coor = self.node_natural(nid).unwrap(); // guaranteed to be valid
            let h_and_grad = self.h_and_grad(NaturalCoor(nat_coor));
            let mats = find_isopar_mats(h_and_grad, self.node_points(), &Self::STRAIN_RULE);

            let c = self.material().get_c(self.material_problem());

            // compute CBU
            let mut st: LinearMatrix = c.mul(&mats.b);
            st = st.mul(&u);
            res.push((nid, StressState::from_vector(st)));
        }

        res
    }

    pub fn jacobian_ratio(&self) -> f64 {
        // returns the ratio of smallest jacobian to largest
        // with an edge case for opposite-sign jacobians
        let samples = NdGaussSamples::new(EL_DIMS, Self::INT_ORDER);
        let jacobians: Vec<f64> = samples
            .map(|(p, _)| {
                let h_and_grad = self.h_and_grad(NaturalCoor(p));
                let mats = find_isopar_mats(h_and_grad, self.node_points(), &Self::STRAIN_RULE);
                mats.det_j
            })
            .collect();

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
}

pub struct ElementMats {
    b: LinearMatrix,
    h: Option<LinearMatrix>,
    j: LinearMatrix,
    j_inv: LinearMatrix,
    det_j: f64,
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

fn find_isopar_mats(
    h_and_grad: [(f64, Point); EL_NODES],
    nodes: [Point; EL_NODES],
    strain_rule: &StrainRule,
) -> ElementMats {
    // initialize matrices for construction
    let mut j = LinearMatrix::zeros((EL_DIMS, EL_DIMS));
    let mut nat_grad_interps = Vec::new();
    for _ in 0..EL_DIMS {
        nat_grad_interps.push(LinearMatrix::zeros((EL_DIMS, EL_DOFS)));
    }
    let mut h_mat = LinearMatrix::zeros((EL_DIMS, EL_DOFS));

    for (i, (h, nat_grad_h)) in h_and_grad.iter().enumerate() {
        let node_loc = nodes[i];
        for x in 0..EL_DIMS {
            // h_mat is a simple arrangement of h values
            h_mat[(x, (EL_DIMS * i) + x)] = *h;
            // partial derivative of h in r/s/t (x is current dimension)
            let nat_grad_in_dim = nat_grad_h[x];

            for y in 0..EL_DIMS {
                // J[x, y] is d (x, y, z)[y] / d (r, s, t)[x]
                // values interpolated from (d h_i / d (r/s/t)) * (x/y/z)_i
                j[(x, y)] += nat_grad_in_dim * node_loc[y];

                // fill out natural (r/s/t) gradient interpolations
                // by the end, ngi[y] * U = grad_rst (u, v, w)[y]
                nat_grad_interps[y][(x, (EL_DIMS * i) + y)] = nat_grad_in_dim;
            }
        }
    }

    // invert J to find the gradient interpolation for the global coordinate system
    let j_inv = j.inverse();
    let det_j = j
        .determinant()
        .expect("unreachable - element determinants should be nonzero");

    // construct the strain interpolation matrix
    let mut b = LinearMatrix::zeros((strain_rule.vec_len(), EL_DOFS));

    for (i, nat_grad_interp) in nat_grad_interps.into_iter().enumerate() {
        // ngi is the natural coordinate gradient interpolation for (u, v, w)[i]
        let grad_interp: LinearMatrix = j_inv.mul(&nat_grad_interp);

        // each row of gi is an interpolation for d(u/v/w)/dx, d(u/v/w)/dy ...
        for j in 0..EL_DIMS {
            // if ngi[j, ..] should be added to a row in b, add it
            if let Some(idx) = strain_rule.dest_idx(i, j) {
                for (k, coeff) in grad_interp.row(j).cloned().enumerate() {
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
