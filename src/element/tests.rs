use super::isopar::{IsoparElement, Bar2Node};
use super::strain::{self, StrainRule};
use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;

#[test]
fn basic_bar_F() {
	let a = Point::new(&[1.5]);
	let b = Point::new(&[2.5]);
	let el = Bar2Node::new(vec![a, b]);
	let target = LinearMatrix::from_flat((2, 2), vec![0.5, -0.5, -0.5, 0.5]);
	let c = LinearMatrix::from_flat((1, 1), vec![1.0]);
	assert_eq!(el.find_k_integrand(Point::new(&[2.3]), &c, strain::Bar), target);
}