use super::isopar::{IsoparElement, Bar2Node, PlaneNNode};
use super::strain::StrainRule;
use super::material::Material;
use super::integrate::{newton_single, nd_gauss_single};
use super::{ElementAssemblage, ElementMap};
use super::loading::Constraint;

use crate::matrix::{LinearMatrix, MatrixLike, Norm};
use crate::spatial::Point;

#[test]
fn basic_bar_f() {
	let a = Point::new(&[1.5]);
	let b = Point::new(&[2.5]);
	let el = Bar2Node::new(vec![a, b]);
	let target = LinearMatrix::from_flat((2, 2), vec![0.5, -0.5, -0.5, 0.5]);
	let c = LinearMatrix::from_flat((1, 1), vec![1.0]);
	assert_eq!(el.find_k_integrand(Point::new(&[2.3]), &c, StrainRule::Bar), target);
}
#[test]
fn rectangle_jacobians() {
	let a = Point::new(&[4.0, 1.0]);
	let b = Point::new(&[-2.0, 1.0]);
	let c = Point::new(&[-2.0, -3.0]);
	let d = Point::new(&[4.0, -3.0]);
	let el = PlaneNNode::new(vec![a, b, c, d]);
	let target = LinearMatrix::from_flat((2, 2), vec![3.0, 0.0, 0.0, 2.0]);

	let mats = el.find_mats(Point::new(&[0.0, 0.0]), StrainRule::PlaneStrain);
	assert!((&mats.j - &target).frobenius() < 1.0e-10);
}
#[test]
fn simple_integrals() {
	let val = newton_single(|x| ((2.0_f64).powf(x) - x), (0.0, 3.0), 0);
	assert!((val - 5.656854).abs() < 1.0e-5);

	let val = nd_gauss_single(|x| x[0].powi(2) * x[1].powi(2), 2, 2);
	assert!(val - (4.0 / 9.0) < 1.0e-5);
}
#[test]
fn assemblage() {
	let mut elas = ElementAssemblage::new(2);
	elas.add_nodes(vec![(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)]);
	elas.add_element(ElementMap::IsoPNN(vec![0, 1, 3, 2]));
	elas.add_conc_force(1, Point::new(&[1.0e6, 0.0]));
	// TODO need better constructors for, like, all of this
	elas.add_constraint(2, Constraint::PlainDof(true, true, false));
	elas.add_constraint(3, Constraint::PlainDof(false, true, false));

	elas.calc_displacements();
	println!("{:?}", elas);
	println!("{:?}", elas.displacements());
	assert!(false);
}