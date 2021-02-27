use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;
use std::convert::TryInto;
use super::material::Material;
use crate::matrix::inverse::Inverse;
use super::strain::StrainRule;

// constructors for isoparametric finite element matrices

pub trait IsoparElement {
    fn new(nodes: Vec<Point>) -> Self;
    fn dim(&self) -> usize;
    fn node_count(&self) -> usize;
    fn node(&self, i: usize) -> Point;
    fn h_and_grad(&self, nat_coor: Point) -> Vec<(f64,  Point)>;

    fn nodes(&self) -> Vec<Point> {
        (0..self.node_count()).map(|x| self.node(x)).collect()
    }

    fn find_mats(&self, nat_coor: Point, strain_rule: StrainRule) -> ElementMats {
        println!("---\nBUILDING ELEMENT MATRICES AT {}", nat_coor);
        print!("{:?}", self.nodes());

        // initialize dim (spatial dimension count) and n (node count)
        let dim = self.dim();
        let n = self.node_count();

        let nodes = self.nodes();

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

        println!("nat_grad_interps:\n{:?}", nat_grad_interps);

        // construct the strain interpolation matrix
        let mut b = LinearMatrix::zeros((strain_rule.vec_len(), dim * n));

        for (i, nat_grad_interp) in nat_grad_interps.into_iter().enumerate() {
            // ngi is the natural coordinate gradient interpolation for (u, v, w)[i]
            let grad_interp: LinearMatrix = j_inv.mul(&nat_grad_interp);
            println!("grad_interp:\n{}", grad_interp);
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

    fn find_k_integrand(&self, nat_coor: Point, c: &LinearMatrix, strain_rule: StrainRule) -> LinearMatrix {
        let mut mats = self.find_mats(nat_coor, strain_rule);

        println!("{}", mats.b);
        print!("{}", mats.j);
        println!("{}", mats.det_j);

        // the integrand for K is (det J) * B_t * C * B
        mats.b.transpose(); // transpose in place is cheap
        let mut inter: LinearMatrix = mats.b.mul(c);
        inter *= mats.det_j;
        mats.b.transpose();

        inter.mul(&mats.b)
    }
}

pub struct Bar2Node {
    nodes: [f64; 2],
}

pub struct ElementMats {
    pub b: LinearMatrix,
    pub h: Option<LinearMatrix>,
    pub j: LinearMatrix,
    pub j_inv: LinearMatrix,
    pub det_j: f64,
}

impl IsoparElement for Bar2Node {
    fn new(nodes: Vec<Point>) -> Bar2Node {
        if nodes.len() != 2 { panic!("bad node count") }
        Bar2Node { nodes: [nodes[0].try_into().unwrap(), nodes[1].try_into().unwrap()] }
    }

    fn dim(&self) -> usize {
        1
    }

    fn node_count(&self) -> usize {
        2
    }

    fn node(&self, i: usize) -> Point {
        self.nodes[i].into()
    }

    fn h_and_grad(&self, nat_coor: Point) -> Vec<(f64, Point)> {
        vec![
            (0.5 * (1.0 - nat_coor[0]), Point::new(&[-0.5])),
            (0.5 * (1.0 + nat_coor[0]), Point::new(&[0.5]))
        ]
    }
}

pub struct PlaneNNode {
    nodes: Vec<(f64, f64)>,
}

impl IsoparElement for PlaneNNode {
    fn new(nodes: Vec<Point>) -> Self {
        // initializes a new planar (1..=9)-node element
        // element should resemble:
        // 1---4---0
        // |   |   |
        // 5---8---7
        // |   |   |
        // 2---6---3
        // degenerate versions are ok, but attempt to preserve node order

        let nodes = nodes.into_iter()
            .map(|p| p.try_into().unwrap())
            .collect();
        PlaneNNode {nodes}
    }

    fn dim(&self) -> usize {
        2
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn node(&self, i: usize) -> Point {
        self.nodes[i].into()
    }

    fn h_and_grad(&self, nat_coor: Point) -> Vec<(f64, Point)> {
        // calculate interpolation functions and their derivatives
        // interpolation is x = sum( h_i(r, s)*x_i )
        // returns (h_i, (dh_i/dr, dh_i/ds)) for i in 0..n
        let (r, s) = nat_coor.try_into().unwrap();

        let mut h = vec![0.0; self.node_count()];
        let mut dhdr = vec![0.0; self.node_count()];
        let mut dhds = vec![0.0; self.node_count()];

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

        if self.dim() <= 4 {
            return h.into_iter()
                .zip(dhdr.into_iter().zip(dhds.into_iter()))
                .map(|(x, y)| (x, y.into()))
                .collect()
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

        if self.dim() <= 5 {
            return h.into_iter()
                .zip(dhdr.into_iter().zip(dhds.into_iter()))
                .map(|(x, y)| (x, y.into()))
                .collect()
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

        if self.dim() <= 6 {
            return h.into_iter()
                .zip(dhdr.into_iter().zip(dhds.into_iter()))
                .map(|(x, y)| (x, y.into()))
                .collect()
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

        if self.dim() <= 7 {
            return h.into_iter()
                .zip(dhdr.into_iter().zip(dhds.into_iter()))
                .map(|(x, y)| (x, y.into()))
                .collect()
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

        if self.dim() <= 7 {
            return h.into_iter()
                .zip(dhdr.into_iter().zip(dhds.into_iter()))
                .map(|(x, y)| (x, y.into()))
                .collect()
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

        return h.into_iter()
                .zip(dhdr.into_iter().zip(dhds.into_iter()))
                .map(|(x, y)| (x, y.into()))
                .collect()
    }
}