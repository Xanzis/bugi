use super::graph::Graph;
use super::inverse::Inverse;
use super::norm::Norm;
use super::{
    CompressedRow, LinearMatrix, LowerRowEnvelope, LowerTriangular, MatrixLike, UpperTriangular,
};

#[test]
fn linear_constructors() {
    // check 'from' constructors against each other
    let a = LinearMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let b = LinearMatrix::from_flat((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(a, b);

    // check 'from' constructors against special ones
    let target = LinearMatrix::from_flat((3, 2), vec![0.0; 6]);
    assert_eq!(target, LinearMatrix::zeros((3, 2)));
    let target = LinearMatrix::from_flat((2, 2), vec![1.0, 0.0, 0.0, 1.0]);
    assert_eq!(target, LinearMatrix::eye(2));

    // check assignment / retrieval in constructed matrices
    let mut x = LinearMatrix::from_flat((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(x[(0, 1)], 2.0);
    assert_eq!(x[(2, 1)], 6.0);
    x[(2, 1)] = 0.0;
    assert_eq!(x[(2, 1)], 0.0);
}

#[test]
fn linear_ops() {
    // check the standard matrix operations for linear matrices
    let a = LinearMatrix::from_flat((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0]);
    let b = LinearMatrix::from_flat((2, 3), vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    let a = &a + &b;
    let target = LinearMatrix::from_flat((2, 3), vec![5.0, 7.0, 9.0, 5.0, 7.0, 10.0]);
    assert_eq!(a, target);

    let a = LinearMatrix::from_flat((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0]);
    let b = LinearMatrix::from_flat(
        (3, 4),
        vec![4.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0],
    );
    let target = LinearMatrix::from_flat((2, 4), vec![9.0, 12.0, 7.0, 9.0, 28.0, 32.0, 20.0, 23.0]);
    let prod: LinearMatrix = a.mul(&b);
    assert_eq!(target, prod);
    let mut a = LinearMatrix::from_flat((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
    a += 2.0;
    let target = LinearMatrix::from_flat((2, 2), vec![3.0, 4.0, 5.0, 6.0]);
    assert_eq!(a, target);
    a *= 2.0;
    let target = LinearMatrix::from_flat((2, 2), vec![6.0, 8.0, 10.0, 12.0]);
    assert_eq!(a, target);
}

#[test]
fn linear_disp() {
    let target = "rows: 1 cols: 2\n1.00000 2.00000 \n";
    let mat = LinearMatrix::from_rows(vec![vec![1.0, 2.0]]);
    println!("{}", mat);
    assert_eq!(format!("{}", mat), target)
}

#[test]
fn linear_transpose() {
    let mut a = LinearMatrix::from_flat((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    a.transpose();
    let b = LinearMatrix::from_flat((3, 2), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(a, b);
}

#[test]
fn linear_swaps() {
    let mut a = LinearMatrix::from_flat((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
    let target_a = LinearMatrix::from_flat((2, 2), vec![3.0, 4.0, 1.0, 2.0]);
    a.swap_rows(0, 1);
    assert_eq!(a, target_a);
    let target_b = LinearMatrix::from_flat((2, 2), vec![4.0, 3.0, 2.0, 1.0]);
    a.swap_cols(0, 1);
    assert_eq!(a, target_b);
}

#[test]
fn iterators() {
    let a = LinearMatrix::from_flat((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(a.row(0).cloned().collect::<Vec<f64>>(), vec![1.0, 2.0]);
    assert_eq!(a.col(0).cloned().collect::<Vec<f64>>(), vec![1.0, 3.0]);
    assert_eq!(
        a.flat().cloned().collect::<Vec<f64>>(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn dots() {
    let a = LinearMatrix::from_flat(2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = LinearMatrix::from_flat(2, vec![2.0, 3.0, 4.0, 5.0]);

    assert_eq!(
        a.row(0).dot(b.col(1)).last().unwrap_or(0.0),
        b.col(1).dot(a.row(0)).last().unwrap_or(0.0)
    );
    assert_eq!(a.row(0).dot(b.col(1)).last().unwrap_or(0.0), 13.0);
}

#[test]
fn solve_gauss() {
    let mut a = LinearMatrix::from_flat((2, 2), vec![1.0, 2.0, 3.0, 4.0]);

    let b = LinearMatrix::from_flat((2, 1), vec![5.0, 6.0]);

    let x = a.solve_gausselim(b).unwrap();

    let target_x = LinearMatrix::from_flat((2, 1), vec![-4.0, 4.5]);
    assert!((&x - &target_x).frobenius() < 1e-10);
}

#[test]
fn lu_decompose() {
    let a = LinearMatrix::from_flat((2, 2), vec![4.0, 3.0, 6.0, 3.0]);
    let (l, u) = a.lu_decompose();
    assert_eq!(l[(0, 0)], 1.0);
    assert_eq!(l[(1, 1)], 1.0);
    assert_eq!(l[(0, 1)], 0.0);
    assert_eq!(u[(1, 0)], 0.0);
    let regen: LinearMatrix = l.mul(&u);
    assert!((&a - &regen).frobenius() < 1e-10);
}

#[test]
fn determinant() {
    let a = LinearMatrix::from_flat((2, 2), vec![3.0, 8.0, 4.0, 6.0]);
    assert!((a.det_lu() + 14.0).abs() < 1e-10);
}

#[test]
fn triangulars() {
    let mut a = LowerTriangular::zeros(3);
    assert_eq!(a.get((1, 2)), Some(&0.0));
    assert_eq!(a.get((3, 2)), None);
    let mut b = UpperTriangular::zeros(3);
    assert_eq!(a.get((1, 2)), Some(&0.0));
    assert_eq!(a.get((3, 2)), None);

    a[(2, 1)] = 3.0;
    assert_eq!(a[(2, 1)], 3.0);
    b[(1, 2)] = 3.0;
    assert_eq!(b[(1, 2)], 3.0);
}

#[test]
#[should_panic]
fn triangular_l_oob() {
    let mut a = LowerTriangular::zeros(2);
    a[(0, 1)] = 2.0;
}

#[test]
#[should_panic]
fn triangular_u_oob() {
    let mut a = UpperTriangular::zeros(2);
    a[(1, 0)] = 2.0;
}

#[test]
fn triangular_l_sub() {
    let mut a = LowerTriangular::eye(4);
    a[(1, 0)] = -1.0;
    a[(2, 1)] = 0.5;
    a[(3, 2)] = 14.0;
    a[(3, 1)] = 1.0;
    a[(3, 0)] = 6.0;
    let x = LinearMatrix::from_flat((4, 1), a.forward_sub(&[1.0, -1.0, 2.0, 1.0]));
    let target = LinearMatrix::from_flat((4, 1), vec![1.0, 0.0, 2.0, -33.0]);
    assert!((&x - &target).frobenius() < 1.0e-10);
}

#[test]
fn triangular_u_sub() {
    let mut a = UpperTriangular::zeros(3);
    a[(0, 0)] = -4.0;
    a[(1, 1)] = 3.5;
    a[(2, 2)] = 3.5;
    a[(0, 1)] = -2.0;
    a[(1, 2)] = -1.75;
    a[(0, 2)] = 1.0;
    let x = LinearMatrix::from_flat((3, 1), a.backward_sub(&[-5.0, 1.75, 10.5]));
    let target = LinearMatrix::from_flat((3, 1), vec![1.0, 2.0, 3.0]);
    assert!((&x - &target).frobenius() < 1.0e-10);
}

#[test]
fn lu_inverses() {
    let mut a = LowerTriangular::eye(3);
    a[(1, 0)] = 3.5;
    a[(2, 2)] = 0.2;
    let b = a.tri_inv();
    let i: LinearMatrix = a.mul(&b);
    let target = LinearMatrix::eye(3);
    assert!((&i - &target).frobenius() < 1.0e-10);

    let mut a = UpperTriangular::eye(3);
    a[(0, 2)] = 0.7;
    a[(1, 1)] = 12.0;
    a[(0, 1)] = 1.0;
    let b = a.tri_inv();
    let i: LinearMatrix = a.mul(&b);
    let target = LinearMatrix::eye(3);
    assert!((&i - &target).frobenius() < 1.0e-10);
}

#[test]
fn inverses() {
    let a = LinearMatrix::from_flat((1, 1), vec![2.0]);
    let a_inv = a.inverse();
    let i: LinearMatrix = a.mul(&a_inv);
    let eye = LinearMatrix::eye(1);
    assert!((&i - &eye).frobenius() < 1.0e-10);

    let a = LinearMatrix::from_flat((2, 2), vec![2.0, 7.6, 3.2, 0.1]);
    let a_inv = a.inverse();
    let i: LinearMatrix = a.mul(&a_inv);
    let eye = LinearMatrix::eye(2);
    assert!((&i - &eye).frobenius() < 1.0e-10);

    let a = LinearMatrix::from_flat((3, 3), vec![2.0, 7.0, 2.5, -1.7, -0.4, 2.3, 1.9, 0.4, 12.0]);
    let a_inv = a.inverse();
    let i: LinearMatrix = a.mul(&a_inv);
    let eye = LinearMatrix::eye(3);
    assert!((&i - &eye).frobenius() < 1.0e-10);
}

#[test]
fn sparse_cr() {
    let a = CompressedRow::from_flat((2, 3), vec![2.0, 0.0, 1.0, 0.0, 3.0, 0.0]);
    let b = CompressedRow::from_flat((3, 1), vec![1.0, 2.0, 0.0]);

    let target = CompressedRow::from_flat((2, 1), vec![2.0, 6.0]);

    let prod: CompressedRow = a.mul(&b);
    assert_eq!(target, prod);
}

#[test]
fn graph_peripheral() {
    let mut g = Graph::from_lol(vec![
        vec![1],
        vec![0, 2, 5],
        vec![1, 3, 5],
        vec![2, 4, 7],
        vec![3, 7],
        vec![1, 2, 6],
        vec![5, 7],
        vec![3, 4, 6],
    ]);

    assert_eq!(g.far_node(2), (0, 2));
    assert_eq!(g.pseudo_peripheral(), 0);
}

#[test]
fn graph_rcm() {
    let mut g = Graph::from_lol(vec![
        vec![1],
        vec![0, 2, 5],
        vec![1, 3, 5],
        vec![2, 4, 7],
        vec![3, 7],
        vec![1, 2, 6],
        vec![5, 7],
        vec![3, 4, 6],
    ]);

    assert_eq!(g.reverse_cuthill_mckee(), vec![7, 4, 6, 3, 5, 2, 1, 0]);
}

#[test]
fn envelope() {
    let envelope = vec![1, 1, 3, 2];
    let mut a = LowerRowEnvelope::from_envelope(envelope);

    a[(0, 0)] = 3.0;
    a[(1, 1)] = 4.0;
    a[(2, 0)] = 5.0;
    a[(2, 1)] = 6.0;
    a[(2, 2)] = 7.0;
    a[(3, 2)] = 8.0;
    a[(3, 3)] = 9.0;

    let target = LowerRowEnvelope::from_flat(
        4,
        vec![
            3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 8.0, 9.0,
        ],
    );

    assert_eq!(a, target)
}

#[test]
fn envelope_l_solve() {
    let mut a = LowerRowEnvelope::from_flat(
        4,
        vec![
            1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 6.0, 1.0, 14.0, 1.0,
        ],
    );
    let x = LinearMatrix::from_flat((4, 1), a.solve(&[1.0, -1.0, 2.0, 1.0]));
    let target = LinearMatrix::from_flat((4, 1), vec![1.0, 0.0, 2.0, -33.0]);
    assert!((&x - &target).frobenius() < 1.0e-10);
}

#[test]
fn envelope_l_solve_transposed() {
    let mut a =
        LowerRowEnvelope::from_flat(3, vec![-4.0, 0.0, 0.0, -2.0, 3.5, 0.0, 1.0, -1.75, 3.5]);

    let x = LinearMatrix::from_flat((3, 1), a.solve_transposed(&[-5.0, 1.75, 10.5]));
    let target = LinearMatrix::from_flat((3, 1), vec![1.0, 2.0, 3.0]);
    assert!((&x - &target).frobenius() < 1.0e-10);
}
