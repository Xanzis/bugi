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
        b.store_polygon(vec![(0.0, 0.0), (2.0, 0.3), (2.0, 5.0), (-1.0, 3.5)]);

        b.divide_all_segments(0.3);

        let mut vis = b.visualize();

        vis.draw("test_generated/chew_bound.png", ());
    }

    #[test]
    fn chew_mesh_init() {
        use super::bounds;
        use super::plane;

        let mut b = bounds::PlaneBoundary::new();
        b.store_polygon(vec![(0.0, 0.0), (2.0, 0.3), (2.0, 5.0), (-1.0, 3.5)]);
        b.divide_all_segments(0.3);

        let mut mesh = plane::PlaneTriangulation::new(b);
        mesh.gift_wrap();

        let mut vis = mesh.visualize();
        vis.draw("test_generated/chew_mesh_init.png", ());
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
}
