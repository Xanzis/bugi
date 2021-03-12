use super::Visualizer;
use crate::spatial::Point;

#[test]
fn one_d() {
    let mut asm: Visualizer = vec![Point::new(&[1.0]), Point::new(&[3.5])].into();
    asm.draw("test_generated/one.png");
}
#[test]
fn two_d() {
    let mut asm: Visualizer = vec![
        Point::new(&[1.0, 2.0]),
        Point::new(&[-25.0, 37.0]),
        Point::new(&[12.0, -5.0]),
    ]
    .into();
    asm.draw("test_generated/two.png");
}
#[test]
fn three_d() {
    let mut asm: Visualizer = vec![
        Point::new(&[-1.0, 1.0, 1.0]),
        Point::new(&[1.0, 1.0, 1.0]),
        Point::new(&[1.0, -1.0, 1.0]),
        Point::new(&[-1.0, -1.0, 1.0]),
        Point::new(&[-1.0, 1.0, -1.0]),
        Point::new(&[1.0, 1.0, -1.0]),
        Point::new(&[1.0, -1.0, -1.0]),
        Point::new(&[-1.0, -1.0, -1.0]),
    ]
    .into();
    asm.edges = Some(vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]);
    asm.draw("test_generated/three.png");
}
#[test]
fn more_points() {
    let mut asm: Visualizer = vec![
        Point::new(&[1.0, 2.0]),
        Point::new(&[-25.0, 37.0]),
        Point::new(&[12.0, -5.0]),
    ]
    .into();

    asm.add_points(
        vec![
            Point::new(&[1.0, 5.0]),
            Point::new(&[-25.0, 40.0]),
            Point::new(&[12.0, -2.0]),
        ],
        1,
    );

    asm.add_points(
        vec![
            Point::new(&[4.0, 2.0]),
            Point::new(&[-22.0, 37.0]),
            Point::new(&[15.0, -5.0]),
        ],
        2,
    );

    asm.draw("test_generated/more.png");
}
