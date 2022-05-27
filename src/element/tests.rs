use super::integrate::{gauss_segment_mat, nd_gauss_single, newton_single};
use super::loading::Constraint;
use super::material::{AL6061, TEST};
use super::{isopar, ElementAssemblage, ElementDescriptor, ElementType};

use crate::matrix::solve::direct::DenseGaussSolver;
use crate::matrix::{LinearMatrix, MatrixLike};
use crate::spatial::Point;

// TODO: reimplement removed element types and move these tests into isopar
// #[test]
// fn basic_bar_f() {
//     let a = Point::new(&[1.5]);
//     let b = Point::new(&[2.5]);
//     let el = IsoparElement::new(&vec![a, b], vec![0, 1], TEST);
//     let target = LinearMatrix::from_flat((2, 2), vec![0.5, -0.5, -0.5, 0.5]);
//     assert_eq!(el.find_k_integrand(Point::new(&[2.3])), target);
// }

// #[test]
// fn rectangle_jacobians() {
//     let a = Point::new(&[4.0, 1.0]);
//     let b = Point::new(&[-2.0, 1.0]);
//     let c = Point::new(&[-2.0, -3.0]);
//     let d = Point::new(&[4.0, -3.0]);
//     let el = IsoparElement::new(&vec![a, b, c, d], vec![0, 1, 2, 3], TEST);
//     //let target = LinearMatrix::from_flat((2, 2), vec![3.0, 0.0, 0.0, 2.0]);

//     let mats = el.find_mats(Point::new(&[0.0, 0.0]));

//     assert_eq!(mats.det_j(), 6.0);
// }

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
fn line_integrals() {
    // segment integral, answer verified by hand
    let integrand = |p: Point| LinearMatrix::from_flat(1, vec![3.0 * p[0].powi(2) - 2.0 * p[1]]);

    let val = gauss_segment_mat(integrand, (3.0, 6.0).into(), (1.0, -1.0).into(), 2);
    assert!((val[(0, 0)] - (8.0 * 53.0_f64.sqrt())).abs() < 1.0e-6);
}

#[test]
fn assemblage() {
    let mut elas = ElementAssemblage::new(2, AL6061);
    elas.set_thickness(0.1);

    let nids = elas.add_nodes(&[(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)]);
    elas.add_element(ElementDescriptor::isopar_triangle([
        nids[0], nids[2], nids[3],
    ]));
    elas.add_element(ElementDescriptor::isopar_triangle([
        nids[3], nids[1], nids[0],
    ]));
    elas.add_conc_force(nids[1], Point::new(&[1.0e7, 0.0]));
    // TODO need better constructors for, like, all of this
    elas.add_constraint(nids[2], Constraint::PlainDof(true, true, false));
    elas.add_constraint(nids[3], Constraint::PlainDof(false, true, false));

    let dfm = elas.calc_displacements::<DenseGaussSolver>();

    let mut vis = dfm.visualize(50.0);
    vis.set_vals(dfm.displacement_norms());
    vis.draw("test_generated/disp_square.png", ());
}

// TODO reimplement rectangular elements
// #[test]
// fn multi_element() {
//     let mut elas = ElementAssemblage::new(2, AL6061);
//     elas.set_thickness(0.1);

//     elas.add_nodes(vec![
//         (0.0, 0.0),
//         (0.1, 0.0),
//         (0.0, 0.1),
//         (0.1, 0.1),
//         (0.0, 0.2),
//         (0.1, 0.2),
//         (0.0, 0.3),
//         (0.1, 0.3),
//         (0.0, 0.4),
//         (0.1, 0.4),
//     ]);
//     for i in 0..4 {
//         let n = 2 * i;
//         elas.add_element(vec![n, n + 1, n + 3, n + 2]);
//     }

//     elas.add_constraint(0, Constraint::PlainDof(true, true, false));
//     elas.add_constraint(1, Constraint::PlainDof(false, true, false));

//     elas.add_conc_force(9, Point::new(&[1.0e5, 0.0]));

//     let dfm = elas.calc_displacements::<DenseGaussSolver>();

//     let mut vis = dfm.visualize(50.0);
//     vis.set_vals(dfm.displacement_norms());
//     vis.draw("test_generated/disp_tower.png", ());
// }

#[test]
fn triangles() {
    let mut elas = ElementAssemblage::new(2, AL6061);
    elas.set_thickness(0.2);

    let mut nids = Vec::new();
    for i in 0..20 {
        let x = i as f64 * 0.2;
        nids.extend(elas.add_nodes(&[(x, 0.0), (x + 0.1, 0.2)]));
    }

    for i in 0..19 {
        let roota = 2 * i;
        let rootb = roota + 1;
        let a_nodes = [nids[roota], nids[roota + 1], nids[roota + 2]];
        let b_nodes = [nids[rootb], nids[rootb + 1], nids[rootb + 2]];
        elas.add_element(ElementDescriptor::isopar_triangle(a_nodes));
        elas.add_element(ElementDescriptor::isopar_triangle(b_nodes));
    }

    elas.add_constraint(nids[0], Constraint::PlainDof(true, true, false));
    elas.add_constraint(nids[29], Constraint::PlainDof(true, true, false));

    elas.add_conc_force(nids[39], Point::new(&[0.0, -1.0e5]));
    let dfm = elas.calc_displacements::<DenseGaussSolver>();

    let mut vis = dfm.visualize(50.0);
    vis.set_vals(dfm.displacement_norms());

    vis.draw("test_generated/triangles_static.png", ());
}

#[test]
fn dist_line() {
    let mut elas = ElementAssemblage::new(2, AL6061);
    elas.set_thickness(0.2);

    let nids = elas.add_nodes(&[(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (1.0, 1.0), (3.0, 1.0)]);

    elas.add_element(ElementDescriptor::isopar_triangle([
        nids[0], nids[1], nids[3],
    ]));
    elas.add_element(ElementDescriptor::isopar_triangle([
        nids[4], nids[3], nids[1],
    ]));
    elas.add_element(ElementDescriptor::isopar_triangle([
        nids[1], nids[2], nids[4],
    ]));

    elas.add_constraint(nids[0], Constraint::PlainDof(false, true, false));
    elas.add_constraint(nids[1], Constraint::PlainDof(true, true, false));
    elas.add_constraint(nids[2], Constraint::PlainDof(false, true, false));

    elas.add_dist_line_force(nids[3], nids[4], Point::new(&[2.0e3, -1.0e3]));

    let dfm = elas.calc_displacements::<DenseGaussSolver>();

    let mut vis = dfm.visualize(100.0);
    vis.set_vals(dfm.displacement_norms());

    vis.draw("test_generated/dist_line.png", ());
}

// TODO this halts on testing for some reason
// #[test]
// fn triangles_modal() {
//     let mut elas = ElementAssemblage::new(2, AL6061);
//     elas.set_thickness(0.2);

//     let mut node_ids = Vec::new();
//     for i in 0..20 {
//         let x = i as f64 * 0.2;
//         node_ids.extend(elas.add_nodes(vec![(x, 0.0), (x + 0.1, 0.2)]));
//     }

//     for i in 0..19 {
//         let roota = 2 * i;
//         let rootb = roota + 1;
//         let a_nodes = [node_ids[roota], node_ids[roota + 1], node_ids[roota + 2]];
//         let b_nodes = [node_ids[rootb], node_ids[rootb + 1], node_ids[rootb + 2]];
//         elas.add_element(ElementDescriptor::isopar_triangle(a_nodes));
//         elas.add_element(ElementDescriptor::isopar_triangle(b_nodes));
//     }

//     elas.add_constraint(node_ids[0], Constraint::PlainDof(true, true, false));
//     elas.add_constraint(node_ids[29], Constraint::PlainDof(true, true, false));

//     let dfms = elas.calc_modes(3);

//     let mut vis = dfms[0].visualize(20.0);
//     vis.set_vals(dfms[0].displacement_norms());

//     vis.draw("test_generated/triangles_modal_1.png", ());

//     let mut vis = dfms[1].visualize(20.0);
//     vis.set_vals(dfms[1].displacement_norms());

//     vis.draw("test_generated/triangles_modal_2.png", ());

//     let mut vis = dfms[2].visualize(20.0);
//     vis.set_vals(dfms[2].displacement_norms());

//     vis.draw("test_generated/triangles_modal_3.png", ());
// }
