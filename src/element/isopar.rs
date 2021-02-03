use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;
use std::convert::TryInto;
use super::material::Material;
use crate::matrix::inverse::Inverse;

// constructors for isoparametric finite element matrices

pub trait IsoparElement {
    fn dim(&self) -> usize;
    fn ncount(&self) -> usize;
    fn new(nodes: Vec<Point>) -> Self;
    fn nodes(&self) -> Vec<Point>;

    fn h_and_grad(&self, nat_coor: Point) -> Vec<(f64, (f64, f64))>;
    fn find_mats(&self, nat_coor: Point) -> ElementMats;

    fn find_K_integrand(&self, nat_coor: Point, c: &LinearMatrix) -> LinearMatrix {
        let mut mats = self.find_mats(nat_coor);

        let mut inter = c.mul(&mats.b);
        inter *= mats.det_j;
        mats.b.transpose(); // modifying b in-place as .transpose() is cheap
        mats.b.mul(&inter)
    }
}

pub struct Bar2Node {
    nodes: [f64; 2],
}

pub struct ElementMats {
    b: LinearMatrix,
    h: Option<LinearMatrix>,
    j: LinearMatrix,
    det_j: f64,
}

impl IsoparElement for Bar2Node {
    fn dim(&self) -> usize {
        1
    }

    fn ncount(&self) -> usize {
        2
    }

    fn new(nodes: Vec<Point>) -> Bar2Node {
        if nodes.len() != 2 { panic!("bad node count") }
        Bar2Node { nodes: [nodes[0].try_into().unwrap(), nodes[1].try_into().unwrap()] }
    }

    fn nodes(&self) -> Vec<Point> {
        vec![self.nodes[0].into(), self.nodes[1].into()]
    }

    fn h_and_grad(&self, nat_coor: Point) -> Vec<(f64, (f64, f64))> {
        unimplemented!()
    }

    fn find_mats(&self, nat_coor: Point) -> ElementMats {
        unimplemented!()
    }

    fn find_K_integrand(&self, nat_coor: Point, c: &LinearMatrix) -> LinearMatrix {
        let nat_coor: f64 = nat_coor.try_into().expect("dimension mismatch");
        let l = self.nodes[1] - self.nodes[0];

        let mut res = LinearMatrix::from_flat((2, 2), vec![1.0, -1.0, -1.0, 1.0]);
        res *= l / 2.0;
        res
    }
}

pub struct PlaneNNode {
    nodes: Vec<(f64, f64)>,
}

impl IsoparElement for PlaneNNode {
    fn dim(&self) -> usize {
        2
    }

    fn ncount(&self) -> usize {
        self.nodes.len()
    }

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

    fn nodes(&self) -> Vec<Point> {
        self.nodes.iter().cloned().map(|x| x.into()).collect()
    }

    fn find_mats(&self, nat_coor: Point) -> ElementMats {
        // assemble the various finite element matrices
        // TODO: have option to only calculate desired matrices
        let interps = self.h_and_grad(nat_coor);
        let n = self.ncount();
        let nodes = self.nodes();

        let mut dxdr = 0.0;
        let mut dydr = 0.0;
        let mut dxds = 0.0;
        let mut dyds = 0.0;
        
        // the following two matrices are interpolation matrices
        // for grad u and grad v, respectively (in r, s)
        let mut u_nat_grad_interp = LinearMatrix::zeros((2, 2 * n));
        let mut v_nat_grad_interp = LinearMatrix::zeros((2, 2 * n));

        // general position / displacement interpolation matrix
        let mut h_mat = LinearMatrix::zeros((2, 2 * n));

        // incorporate the gradient of each interpolation function 
        for (i, (h, (dhdr, dhds))) in interps.into_iter().enumerate() {
            let node = nodes[i];
            let x = node[0];
            let y = node[1];

            // first row of j is dx/dr, dy/dr
            dxdr += x * dhdr;
            dydr += y * dhdr;
            // second row is dx/ds, dy/ds
            dxds += x * dhds;
            dyds += y * dhds;

            u_nat_grad_interp.put((0, 2 * i), dhdr);
            u_nat_grad_interp.put((1, 2 * i), dhds);

            v_nat_grad_interp.put((0, (2 * i) + 1), dhdr);
            v_nat_grad_interp.put((1, (2 * i) + 1), dhds);

            h_mat.put((0, 2 * i), h);
            h_mat.put((0, (2 * i) + 1), h);
        }

        // construct J, invert it, get its determinant
        let j = LinearMatrix::from_flat((2, 2), vec![dxdr, dydr, dxds, dyds]);
        let det_j = j.determinant();
        let j_inv = j.inverse();

        // assemble x/y gradient interpolation matrices
        let u_grad_interp = j_inv.mul(&u_nat_grad_interp);
        let v_grad_interp = j_inv.mul(&v_nat_grad_interp);

        // find strain interpolations (row vectors)
        let e_xx: Vec<f64> = u_grad_interp.row(0).cloned().collect();
        let e_yy: Vec<f64> = v_grad_interp.row(1).cloned().collect();
        let e_xy: Vec<f64> = u_grad_interp.row(1)
            .zip(v_grad_interp.row(0))
            .map(|(x, y)| x + y)
            .collect();

        let b = LinearMatrix::from_rows(vec![e_xx, e_yy, e_xy]);

        ElementMats { b, h: None, j, det_j }
    }

    fn h_and_grad(&self, nat_coor: Point) -> Vec<(f64, (f64, f64))> {
        // calculate interpolation functions and their derivatives
        // interpolation is x = sum( h_i(r, s)*x_i )
        // returns (h_i, (dh_i/dr, dh_i/ds)) for i in 0..n
        let (r, s) = nat_coor.try_into().unwrap();

        let mut h = vec![0.0; self.dim()];
        let mut dhdr = vec![0.0; self.dim()];
        let mut dhds = vec![0.0; self.dim()];

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
            return h.into_iter().zip(dhdr.into_iter().zip(dhds.into_iter())).collect()
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
            return h.into_iter().zip(dhdr.into_iter().zip(dhds.into_iter())).collect()
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
            return h.into_iter().zip(dhdr.into_iter().zip(dhds.into_iter())).collect()
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
            return h.into_iter().zip(dhdr.into_iter().zip(dhds.into_iter())).collect()
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
            return h.into_iter().zip(dhdr.into_iter().zip(dhds.into_iter())).collect()
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

        h.into_iter().zip(dhdr.into_iter().zip(dhds.into_iter())).collect()
    }
}