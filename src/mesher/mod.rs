use std::error;
use std::fmt;

pub mod bounds;
pub mod plane;

#[derive(Clone, Debug)]
pub enum MeshError {
    TriangleError(String),
}

impl MeshError {
    fn triangle<T: ToString>(msg: T) -> Self {
        MeshError::TriangleError(msg.to_string())
    }
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeshError::TriangleError(s) => write!(f, "TriangleError: {}", s),
        }
    }
}

impl error::Error for MeshError {}

#[cfg(test)]
mod tests {
    #[test]
    fn chew_bound() {
        use super::bounds;

        let mut b = bounds::PlaneBoundary::new();
        let poly: [(f64, f64); 8] = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 3.0),
            (-5.0, 2.5),
            (-5.0, 1.5),
            (1.0, 2.0),
            (1.0, 1.0),
            (0.0, 1.5),
        ];
        b.store_polygon(poly.iter().map(|p| p.clone()).collect::<Vec<_>>());

        b.divide_all_segments(0.45);

        let mut vis = b.visualize();

        vis.draw("test_generated/chew_bound.png", ());
    }

    #[test]
    fn chew_mesh() {
        use super::bounds;
        use super::plane;

        let mut b = bounds::PlaneBoundary::new();
        b.store_polygon(vec![(0.0, 0.0), (2.0, 0.3), (2.0, 5.0), (-1.0, 3.5)]);

        let mut mesh = plane::PlaneTriangulation::new(b);
        mesh.chew_mesh(0.3);

        let mut vis = mesh.visualize();
        vis.draw("test_generated/chew_mesh.png", ());
    }

    #[test]
    fn big_mesh() {
        use super::bounds;
        use super::plane;

        let mut b = bounds::PlaneBoundary::new();

        let poly: [(f64, f64); 44] = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 3.0),
            (2.0, 2.0),
            (3.0, 3.0),
            (3.0, 0.0),
            (12.0, 0.0),
            (12.0, 2.0),
            (13.0, 2.0),
            (13.0, 0.0),
            (14.0, 0.0),
            (14.0, 5.0),
            (13.0, 5.0),
            (13.0, 3.0),
            (12.0, 3.0),
            (12.0, 5.0),
            (11.0, 5.0),
            (11.0, 1.0),
            (10.0, 1.0),
            (10.0, 3.0),
            (9.0, 3.0),
            (9.0, 4.0),
            (10.0, 4.0),
            (10.0, 5.0),
            (8.0, 5.0),
            (8.0, 2.0),
            (9.0, 2.0),
            (9.0, 1.0),
            (6.0, 1.0),
            (6.0, 2.0),
            (7.0, 2.0),
            (7.0, 3.0),
            (6.0, 3.0),
            (6.0, 4.0),
            (7.0, 4.0),
            (7.0, 5.0),
            (5.0, 5.0),
            (5.0, 1.0),
            (4.0, 1.0),
            (4.0, 5.0),
            (3.0, 5.0),
            (2.0, 4.0),
            (1.0, 5.0),
            (0.0, 5.0),
        ];

        b.store_polygon(poly.iter().map(|p| p.clone()).collect::<Vec<_>>());

        let mut mesh = plane::PlaneTriangulation::new(b);
        mesh.chew_mesh(0.3);

        let mut vis = mesh.visualize();
        vis.draw("test_generated/chew_big_mesh.png", vec!["im_size=1024"]);
    }
}
