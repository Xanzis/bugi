use super::integrate::{self, GAUSS_TRI_SAMPLES};
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

        // TODO switch this over to using naturalcoor

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

    fn get_isopar_mats(&self, c: NaturalCoor) -> ElementMats {
        // formulate the finite element matrices at the given natural coordinate
        // for the given strain condition
        // TODO switch over to spacemath Point for this mod
        let (r, s) = c.0.try_into().expect("bad point input");

        // compute the interpolation coefficients h_i and their derivatives
        let h = [1.0 - r - s, r, s];
        let dhdr = [-1.0, 1.0, 0.0];
        let dhds = [-1.0, 0.0, 1.0];
        let h_grads = LinearMatrix::from_flat(
            (EL_DIMS, EL_NODES),
            [dhdr[0], dhdr[1], dhdr[2], dhds[0], dhds[1], dhds[2]],
        );
        // h_grads is now [dh1/dr dh2/dr dh3/dr]
        //                [dh1/ds dh2/ds dh3/ds]

        // construct finite element matrices
        // H is formulated such that u(x, y) = H(r, s) . U
        // where U is a vector of each element DoF's displacement

        let h_mat = LinearMatrix::from_flat(
            (EL_DIMS, EL_DOFS),
            [
                h[0], 0.0, h[1], 0.0, h[2], 0.0, 0.0, h[0], 0.0, h[1], 0.0, h[2],
            ],
        );

        // construct the jacobian of the natural coordinate transformation
        // J = [dx/dr dy/dr]
        //     [dx/ds dy/ds]
        let node_points_mat = LinearMatrix::from_points_row(self.node_points().to_vec()); // TODO precompute this, stick in const matrix
        let j: LinearMatrix = h_grads.mul(&node_points_mat); // this is proooobably it
        assert_eq!(j.shape(), (EL_DIMS, EL_DIMS)); // TODO remove once confirmed

        // invert J to find the gradient interpolation for the global coordinate system
        let j_inv = j.inverse();
        let det_j = j
            .determinant()
            .expect("unreachable - element determinants should be nonzero");

        // find the global coordinates, which are needed for some strain rules
        let coords: LinearMatrix = LinearMatrix::row_vec(h.to_vec()).mul(&node_points_mat);
        let coords = (coords[(0, 0)], coords[(0, 1)]);

        // construct B, first finding the gradient of h in the global coordinate system
        let h_grads_global: LinearMatrix = j_inv.mul(&h_grads);
        // h_grads_global is now [dh1/dx dh2/dx dh3/dx]
        //                       [dh1/dy dh2/dy dh3/dy]

        let b_mat = Self::STRAIN_RULE.build_b(&h_grads_global, coords.into()); // todo switch away from constant strain rule

        ElementMats {
            b: b_mat,
            h: h_mat,
            j,
            j_inv,
            det_j,
        }
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
            let mut mats = self.get_isopar_mats(NaturalCoor(nat_coor));

            // the integrand for K is thickness * (det J) * B_t * C * B
            mats.b.transpose(); // transpose in place is cheap
            let mut inter: LinearMatrix = mats.b.mul(&c);
            inter *= mats.det_j;
            mats.b.transpose();

            inter.mul(&mats.b)
        };

        let mut el_k: LinearMatrix = integrate::gauss_tri(integrand);
        el_k *= self.thickness;

        // process the result for handoff to the element assemblage
        self.map_matrix(&el_k)
    }

    pub fn calc_m(&self) -> Vec<(NodeDof, NodeDof, f64)> {
        let integrand = |nat_coor| {
            let mats = self.get_isopar_mats(NaturalCoor(nat_coor));

            // integrand is rho * scaling * H_t * H
            let h = mats.h;
            let mut ht = h.clone();
            ht.transpose();
            ht *= self.material().density() * self.thickness;

            ht.mul(&h)
        };

        let el_m: LinearMatrix = integrate::gauss_tri(integrand);

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

        let integrand = |nat_coor| {
            let mats = self.get_isopar_mats(NaturalCoor(nat_coor));
            let mut h = mats.h;
            h.transpose();
            // h *= mats.det_j; TODO check textbook
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
        let res = Some(self.map_vec(&res));
        res
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
            let mats = self.get_isopar_mats(NaturalCoor(nat_coor));

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
        let jacobians: Vec<f64> = GAUSS_TRI_SAMPLES
            .iter()
            .map(|&p| {
                let mats = self.get_isopar_mats(NaturalCoor(p.into()));
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

#[derive(Debug, Clone)]
pub struct ElementMats {
    b: LinearMatrix,
    h: LinearMatrix,
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

    pub fn get_h<'a>(&'a self) -> &'a LinearMatrix {
        &self.h
    }
}
