use super::isopar::IsoparElement;
use super::strain::StrainRule;
use super::material::{Material, AL6061, TEST};
use super::integrate::{newton_single, nd_gauss_single};
use super::ElementAssemblage;
use super::loading::Constraint;

use crate::matrix::{LinearMatrix, MatrixLike, Norm};
use crate::spatial::Point;

#[test]
fn basic_bar_f() {
	let a = Point::new(&[1.5]);
	let b = Point::new(&[2.5]);
	let el = IsoparElement::new(&vec![a, b], vec![0, 1], TEST);
	let target = LinearMatrix::from_flat((2, 2), vec![0.5, -0.5, -0.5, 0.5]);
	assert_eq!(el.find_k_integrand(Point::new(&[2.3])), target);
}
#[test]
fn rectangle_jacobians() {
	let a = Point::new(&[4.0, 1.0]);
	let b = Point::new(&[-2.0, 1.0]);
	let c = Point::new(&[-2.0, -3.0]);
	let d = Point::new(&[4.0, -3.0]);
	let el = IsoparElement::new(&vec![a, b, c, d], vec![0, 1, 2, 3], TEST);
	let target = LinearMatrix::from_flat((2, 2), vec![3.0, 0.0, 0.0, 2.0]);

	let mats = el.find_mats(Point::new(&[0.0, 0.0]));
	assert!((mats.j() - &target).frobenius() < 1.0e-10);
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
	use crate::visual::Visualizer;

	let mut elas = ElementAssemblage::new(2, AL6061);
	elas.set_thickness(0.1);

	elas.add_nodes(vec![(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)]);
	elas.add_element(vec![0, 2, 3, 1]);
	elas.add_conc_force(1, Point::new(&[1.0e7, 0.0]));
	// TODO need better constructors for, like, all of this
	elas.add_constraint(2, Constraint::PlainDof(true, true, false));
	elas.add_constraint(3, Constraint::PlainDof(false, true, false));

	elas.calc_displacements();
	
	let mut vis: Visualizer = elas.nodes().clone().into();
	vis.add_points(elas.displaced_nodes(50.0).unwrap(), 1);

	vis.draw("test_generated/disp_square.png");
}
#[test]
fn multi_element() {
	use crate::visual::Visualizer;

	let mut elas = ElementAssemblage::new(2, AL6061);
	elas.set_thickness(0.1);

	elas.add_nodes(vec![
		(0.0, 0.0),
		(0.1, 0.0),
		(0.0, 0.1),
		(0.1, 0.1),
		(0.0, 0.2),
		(0.1, 0.2),
		(0.0, 0.3),
		(0.1, 0.3),
		(0.0, 0.4),
		(0.1, 0.4)]);
	for i in 0..4 {
		let n = 2 * i;
		elas.add_element(vec![n, n + 1, n + 3, n + 2]);
	}
	
	elas.add_constraint(0, Constraint::PlainDof(true, true, false));
	elas.add_constraint(1, Constraint::PlainDof(false, true, false));

	elas.add_conc_force(9, Point::new(&[1.0e5, 0.0]));

	elas.calc_displacements();

	println!("{:?}", elas.displacements());

	let mut vis: Visualizer = elas.nodes().clone().into();
	vis.add_points(elas.displaced_nodes(50.0).unwrap(), 1);

	vis.draw("test_generated/disp_tower.png");
}