use super::integrate::{nd_gauss_single, newton_single};
use super::isopar::IsoparElement;
use super::loading::Constraint;
use super::material::{AL6061, TEST};
use super::ElementAssemblage;

use crate::matrix::{LinearMatrix, MatrixLike};
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
    //let target = LinearMatrix::from_flat((2, 2), vec![3.0, 0.0, 0.0, 2.0]);

    let mats = el.find_mats(Point::new(&[0.0, 0.0]));

    assert_eq!(mats.det_j(), 6.0);
}
#[test]
fn simple_integrals() {
    // single integral of (2^x - x) with Newton samples
    let val = newton_single(|x| ((2.0_f64).powf(x) - x), (0.0, 3.0), 0);
    assert!((val - 5.656854).abs() < 1.0e-5);

    // single integral of (x^2) with gauss samples
    let val = nd_gauss_single(|x| x[0].powi(2), 1, 2);
    assert!((val - (2.0 / 3.0)).abs() < 1.0e-6);

    // double integral of (x^2 * y^2) with gauss samples
    let val = nd_gauss_single(|x| x[0].powi(2) * x[1].powi(2), 2, 2);
    assert!((val - (4.0 / 9.0)).abs() < 1.0e-6);

    // triple integral of (x * y^2 * z^3) with gauss samples
    let val = nd_gauss_single(|x| x[0].powi(2) * x[1].powi(2) * x[2].powi(4), 3, 4);
    assert!((val - (8.0 / 45.0)).abs() < 1.0e-6);
}
#[test]
fn assemblage() {
    let mut elas = ElementAssemblage::new(2, AL6061);
    elas.set_thickness(0.1);

    elas.add_nodes(vec![(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)]);
    elas.add_element(vec![0, 2, 3, 1]);
    elas.add_conc_force(1, Point::new(&[1.0e7, 0.0]));
    // TODO need better constructors for, like, all of this
    elas.add_constraint(2, Constraint::PlainDof(true, true, false));
    elas.add_constraint(3, Constraint::PlainDof(false, true, false));

    elas.calc_displacements();

    let mut vis = elas.visualize(50.0);
    vis.set_vals(elas.displacement_norms().unwrap());
    vis.draw("test_generated/disp_square.png", ());
}
#[test]
fn multi_element() {
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
        (0.1, 0.4),
    ]);
    for i in 0..4 {
        let n = 2 * i;
        elas.add_element(vec![n, n + 1, n + 3, n + 2]);
    }

    elas.add_constraint(0, Constraint::PlainDof(true, true, false));
    elas.add_constraint(1, Constraint::PlainDof(false, true, false));

    elas.add_conc_force(9, Point::new(&[1.0e5, 0.0]));

    elas.calc_displacements();

    let mut vis = elas.visualize(50.0);
    vis.set_vals(elas.displacement_norms().unwrap());
    vis.draw("test_generated/disp_tower.png", ());
}
#[test]
fn triangles() {
    let mut elas = ElementAssemblage::new(2, AL6061);
    elas.set_thickness(0.2);

    for i in 0..20 {
        let x = i as f64 * 0.2;
        elas.add_nodes(vec![(x, 0.0), (x + 0.1, 0.2)]);
    }

    for i in 0..19 {
        let roota = 2 * i;
        let rootb = roota + 1;
        elas.add_element(vec![roota, roota + 1, roota + 2]);
        elas.add_element(vec![rootb, rootb + 1, rootb + 2]);
    }

    elas.add_constraint(0, Constraint::PlainDof(true, true, false));
    elas.add_constraint(29, Constraint::PlainDof(false, true, false));

    elas.add_conc_force(39, Point::new(&[0.0, -1.0e5]));
    elas.calc_displacements();

    let mut vis = elas.visualize(50.0);
    vis.set_vals(elas.displacement_norms().unwrap());

    vis.draw("test_generated/triangles.png", ());
}
