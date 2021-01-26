use super::{LinearMatrix, MatrixLike};
use super::inverse::Inverse;
use super::norm::Norm;
#[test]
fn linear_constructors() {
    let target = LinearMatrix::from_flat((3, 2).into(), vec![0.0; 6]);
    assert_eq!(target, LinearMatrix::zeros((3, 2).into()));
    let target = LinearMatrix::from_flat((2, 2).into(), vec![1.0, 0.0, 0.0, 1.0]);
    assert_eq!(target, LinearMatrix::eye(2));
}
#[test]
fn matadd() {
    let mut a = LinearMatrix::from_flat((2, 3).into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0]);
    let b = LinearMatrix::from_rows(vec![vec![4.0, 5.0, 6.0], vec![1.0, 2.0, 3.0]]);
    a.add_ass(&b);
    let target = LinearMatrix::from_rows(vec![vec![5.0, 7.0, 9.0], vec![5.0, 7.0, 10.0]]);
    assert_eq!(a, target);
}
#[test]
fn matmul() {
    let a = LinearMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 7.0]]);
    let b = LinearMatrix::from_rows(vec![
        vec![4.0, 2.0, 2.0, 1.0],
        vec![1.0, 2.0, 1.0, 1.0],
        vec![1.0, 2.0, 1.0, 2.0],
    ]);
    let target = LinearMatrix::from_rows(vec![
        vec![9.0, 12.0, 7.0, 9.0],
        vec![28.0, 32.0, 20.0, 23.0],
    ]);
    let prod = a.mul(&b);
    assert_eq!(target, prod);
}
#[test]
fn disp() {
    let target = "rows: 1 cols: 2\n1.00000 2.00000 \n";
    let mat = LinearMatrix::from_rows(vec![vec![1.0, 2.0]]);
    println!("{}", mat);
    assert_eq!(format!("{}", mat), target)
}
#[test]
fn transpose() {
    let mut a = LinearMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    a.transpose();
    let b = LinearMatrix::from_rows(vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    assert_eq!(a, b);
}
#[test]
fn swaps() {
    let mut a = LinearMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let target_a = LinearMatrix::from_rows(vec![vec![3.0, 4.0], vec![1.0, 2.0]]);
    a.swap_rows(0, 1);
    assert_eq!(a, target_a);
    let target_b = LinearMatrix::from_rows(vec![vec![4.0, 3.0], vec![2.0, 1.0]]);
    a.swap_cols(0, 1);
    assert_eq!(a, target_b);
}
#[test]
fn solve_gauss() {
    let mut a = LinearMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = LinearMatrix::from_rows(vec![vec![5.0], vec![6.0]]);

    let mut x = a.solve_gausselim(b).unwrap();

    let target_x = LinearMatrix::from_rows(vec![vec![-4.0], vec![4.5]]);
    x.sub_ass(&target_x);
    assert!(x.frobenius() < 1e-10);
}