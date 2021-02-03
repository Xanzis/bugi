use super::isopar::{IsoparElement, Bar2Node};
use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;

#[test]
fn basic_bar_F() {
	let a = Point::One(1.0);
	let b = Point::One(3.0);
	let el = Bar2Node::new(vec![a, b]);
	let target = LinearMatrix::from_flat((2, 2), vec![1.0, -1.0, -1.0, 1.0]);
	let c = LinearMatrix::from_flat((1, 1), vec![1.0]);
	assert_eq!(el.find_K_integrand(Point::One(2.3), &c), target);
}