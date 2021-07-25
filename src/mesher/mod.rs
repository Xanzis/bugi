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

        let poly = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 3.0),
            (2.0, 2.0),
            (3.0, 3.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (4.0, 1.0),
            (4.0, 5.0),
            (3.0, 5.0),
            (2.0, 4.0),
            (1.0, 5.0),
            (0.0, 5.0),
        ];

        b.store_polygon(poly);

        let mut mesh = plane::PlaneTriangulation::new(b);
        mesh.chew_mesh(0.5);

        let mut vis = mesh.visualize();
        vis.draw("test_generated/chew_big_mesh.png", vec!["im_size=1024"]);
    }

    #[test]
    fn distributed() {
        use super::bounds;
        use crate::element::loading::Constraint;

        let mut bound = bounds::PlaneBoundary::new();

        // store vertices, get vertex ids, store segments
        let a = bound.store_vertex((0.0, 0.0));
        let b = bound.store_vertex((1.0, 0.0));
        let c = bound.store_vertex((1.0, 1.0));
        let d = bound.store_vertex((0.0, 1.0));

        bound.store_segment((a, b));
        bound.store_segment((b, c));
        bound.store_segment((c, d));
        bound.store_segment((d, a));

        // store a distributed constraint and a distributed force
        bound.store_distributed_constraint(a, b, Constraint::PlainDof(true, false, false));
        bound.store_distributed_force(c, d, (1.0, 2.0).into());

        // get the boundary to subdivide every edge
        bound.divide_all_segments(0.49);

        // check that the point (0.5, 0.0) has been created and has an associated constraint
        let has_con = bound
            .all_constraints()
            .into_iter()
            .map(|(vid, _)| bound.get(vid).unwrap())
            .any(|x| x == (0.5, 0.0));
        assert!(has_con);

        // check that a new segment from c to a point at (0.5, 1.0) exists and has a force
        let fwd = bound
            .all_distributed_forces()
            .into_iter()
            .map(|(s, _)| (s.0, bound.get(s.1).unwrap()))
            .any(|(sa, sb_val)| sa == c && sb_val == (0.5, 1.0));
        let rev = bound
            .all_distributed_forces()
            .into_iter()
            .map(|(s, _)| (s.1, bound.get(s.0).unwrap()))
            .any(|(sa, sb_val)| sa == c && sb_val == (0.5, 1.0));
        assert!(fwd || rev);
    }
}
