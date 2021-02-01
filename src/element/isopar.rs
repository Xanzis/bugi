use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;
use std::convert::TryInto;

// constructors for isoparametric finite element matrices

pub trait IsoparElement {
	fn dim(&self) -> usize;
	fn new(nodes: Vec<Point>) -> Self;
	fn get_F(&self, nat_coor: Point) -> LinearMatrix;
}

pub struct Bar2Node {
	nodes: [f64; 2],
}

impl IsoparElement for Bar2Node {
	fn dim(&self) -> usize {
		1
	}

	fn new(nodes: Vec<Point>) -> Bar2Node {
		if nodes.len() != 2 { panic!("bad node count") }
		Bar2Node { nodes: [nodes[0].try_into().unwrap(), nodes[1].try_into().unwrap()] }
	}

	fn get_F(&self, nat_coor: Point) -> LinearMatrix {
		let nat_coor: f64 = nat_coor.try_into().expect("dimension mismatch");
		let l = self.nodes[1] - self.nodes[0];

		let mut res = LinearMatrix::from_flat((2, 2), vec![1.0, -1.0, -1.0, 1.0]);
		res *= l / 2.0;
		res
	}
}