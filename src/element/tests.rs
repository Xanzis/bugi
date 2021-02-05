use super::isopar::{IsoparElement, Bar2Node, PlaneNNode};
use super::strain::{self, StrainRule};
use super::material::{Aluminum6061, ProblemType, Material};
use super::integrate::{newton_single, nd_gauss_single};
use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;

#[test]
fn basic_bar_f() {
	let a = Point::new(&[1.5]);
	let b = Point::new(&[2.5]);
	let el = Bar2Node::new(vec![a, b]);
	let target = LinearMatrix::from_flat((2, 2), vec![0.5, -0.5, -0.5, 0.5]);
	let c = LinearMatrix::from_flat((1, 1), vec![1.0]);
	assert_eq!(el.find_k_integrand(Point::new(&[2.3]), &c, strain::Bar), target);
}
#[test]
fn rectangle_jacobians() {
	let a = Point::new(&[4.0, 1.0]);
	let b = Point::new(&[-2.0, 1.0]);
	let c = Point::new(&[-2.0, -3.0]);
	let d = Point::new(&[4.0, -3.0]);
	let el = PlaneNNode::new(vec![a, b, c, d]);
	let target = LinearMatrix::from_flat((2, 2), vec![3.0, 0.0, 0.0, 2.0]);

	let mats = el.find_mats(Point::new(&[0.0, 0.0]), strain::PlaneStrain);
	assert_eq!(mats.j, LinearMatrix::from_flat((2, 2), vec![3.0, 0.0, 0.0, 2.0]));
}
#[test]
fn simple_integrals() {
	let val = newton_single(|x| ((2.0_f64).powf(x) - x), (0.0, 3.0), 0);
	assert!((val - 5.656854).abs() < 1.0e-5);

	let val = nd_gauss_single(|x| x[0].powi(2) * x[1].powi(2), 2, 0);
	assert!(val - (4.0 / 9.0) < 1.0e-5);
}