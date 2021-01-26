use super::{LinearMatrix, MatrixLike};
use super::inverse::Inverse;
use super::norm::Norm;

#[test]
fn linear_constructors() {
    // check 'from' constructors against each other
    let a = LinearMatrix::from_rows(vec![vec![1.0, 2.0, 3.0],
                                         vec![4.0, 5.0, 6.0]]);
    let b = LinearMatrix::from_flat((2, 3), vec![1.0, 2.0, 3.0, 
                                                 4.0, 5.0, 6.0]);
    assert_eq!(a, b);

    // check 'from' constructors against special ones
    let target = LinearMatrix::from_flat((3, 2), vec![0.0; 6]);
    assert_eq!(target, LinearMatrix::zeros((3, 2)));
    let target = LinearMatrix::from_flat((2, 2), vec![1.0, 0.0,
                                                      0.0, 1.0]);
    assert_eq!(target, LinearMatrix::eye(2));

    // check assignment / retrieval in constructed matrices
    let mut x = LinearMatrix::from_flat((3, 2), vec![1.0, 2.0, 
                                                     3.0, 4.0, 
                                                     5.0, 6.0]);
    assert_eq!(x.get((0, 1)), Some(&2.0));
    assert_eq!(x.get((2, 1)), Some(&6.0));
    x.put((2, 1), 0.0);
    assert_eq!(x.get((2, 1)), Some(&0.0));
}
#[test]
fn linear_ops() {
    // check the standard matrix operations for linear matrices
    let mut a = LinearMatrix::from_flat((2, 3), vec![1.0, 2.0, 3.0,
                                                     4.0, 5.0, 7.0]);
    let b = LinearMatrix::from_flat((2, 3), vec![4.0, 5.0, 6.0, 
                                                 1.0, 2.0, 3.0]);
    let a = &a + &b;
    let target = LinearMatrix::from_flat((2, 3), vec![5.0, 7.0, 9.0,
                                                      5.0, 7.0, 10.0]);
    assert_eq!(a, target);

    let a = LinearMatrix::from_flat((2, 3), vec![1.0, 2.0, 3.0, 
                                               4.0, 5.0, 7.0]);
    let b = LinearMatrix::from_flat((3, 4), vec![4.0, 2.0, 2.0, 1.0, 
                                                 1.0, 2.0, 1.0, 1.0, 
                                                 1.0, 2.0, 1.0, 2.0]);
    let target = LinearMatrix::from_flat((2, 4), vec![9.0, 12.0, 7.0, 9.0, 
                                                      28.0, 32.0, 20.0, 23.0]);
    let prod = a.mul(&b);
    assert_eq!(target, prod);
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
    let mut a = LinearMatrix::from_flat((2, 3), vec![1.0, 2.0, 3.0, 
                                                     4.0, 5.0, 6.0]);
    a.transpose();
    let b = LinearMatrix::from_flat((3, 2), vec![1.0, 4.0, 
                                                 2.0, 5.0, 
                                                 3.0, 6.0]);
    assert_eq!(a, b);
}
#[test]
fn linear_swaps() {
    let mut a = LinearMatrix::from_flat((2, 2), vec![1.0, 2.0, 
                                                     3.0, 4.0]);

    let target_a = LinearMatrix::from_flat((2, 2), vec![3.0, 4.0, 
                                                        1.0, 2.0]);
    a.swap_rows(0, 1);
    assert_eq!(a, target_a);
    let target_b = LinearMatrix::from_flat((2, 2), vec![4.0, 3.0, 
                                                        2.0, 1.0]);
    a.swap_cols(0, 1);
    assert_eq!(a, target_b);
}
#[test]
fn solve_gauss() {
    let mut a = LinearMatrix::from_flat((2, 2), vec![1.0, 2.0, 
                                                     3.0, 4.0]);

    let b = LinearMatrix::from_flat((2, 1), vec![5.0, 
                                                 6.0]);

    let mut x = a.solve_gausselim(b).unwrap();

    let target_x = LinearMatrix::from_flat((2, 1), vec![-4.0, 
                                                        4.5]);
    assert!((&x - &target_x).frobenius() < 1e-10);
}