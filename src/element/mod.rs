pub mod constraint;
pub mod element;
pub mod integrate;
pub mod material;
pub mod strain;
pub mod stress;

use crate::matrix::solve::eigen::{DeterminantSearcher, EigenSystem};
use crate::matrix::solve::{Solver, System};
use crate::matrix::{Average, Dictionary, LinearMatrix, MatrixLike};
use crate::spatial::Point;
use crate::visual::Visualizer;

use constraint::{Constraint, DofTransform};
use element::Element;
use material::Material;
use strain::Condition;
use stress::StressState;

use std::collections::HashMap;

#[cfg(test)]
mod tests;

const NODE_DIM: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub fn new(i: usize) -> Self {
        Self(i)
    }

    pub fn into_idx(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Dof(usize);

impl Dof {
    fn to_node_dof(self) -> NodeDof {
        NodeDof(NodeId(self.0 / NODE_DIM), self.0 % NODE_DIM)
    }
}

// a node idx + dof of that node (x/y/z) pair
#[derive(Debug, Clone, Copy)]
pub struct NodeDof(NodeId, usize);

impl NodeDof {
    fn to_dof(self) -> Dof {
        Dof(self.0 .0 * NODE_DIM + self.1)
    }
}

#[derive(Debug, Clone)]
pub struct ElementDescriptor {
    nodes: [NodeId; 3],
}

impl ElementDescriptor {
    pub fn new(nodes: [NodeId; 3]) -> Self {
        Self { nodes }
    }

    pub fn into_parts(self) -> [NodeId; 3] {
        self.nodes
    }
}

fn build_element(elas: &ElementAssemblage, desc: ElementDescriptor) -> Element {
    Element::new(elas, desc.nodes)
}

// the big one - primary puclic-facing API for this crate
#[derive(Debug)]
pub struct ElementAssemblage {
    nodes: Vec<Point>,

    condition: Condition,
    material: Material,

    elements: Vec<Element>,
    constraints: Vec<Constraint>,
    concentrated_forces: HashMap<NodeId, Point>,
    line_forces: HashMap<(NodeId, NodeId), Point>,
}

impl ElementAssemblage {
    pub fn new(material: Material, condition: Condition) -> Self {
        ElementAssemblage {
            nodes: Vec::new(),
            elements: Vec::new(),
            constraints: Vec::new(),
            concentrated_forces: HashMap::new(),
            line_forces: HashMap::new(),
            condition,
            material,
        }
    }

    pub fn condition(&self) -> Condition {
        self.condition
    }

    pub fn material(&self) -> Material {
        self.material
    }

    pub fn node(&self, n: NodeId) -> Point {
        // TODO maybe make a non-panicking version
        self.nodes[n.0]
    }

    pub fn nodes(&self) -> Vec<Point> {
        self.nodes.clone()
    }

    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        // return an iterator over all currently valid node ids
        (0..self.nodes.len()).map(|x| NodeId(x))
    }

    #[must_use = "nodes can only be referenced by their assigned NodeId"]
    pub fn add_nodes<T: Into<Point> + Clone>(&mut self, ns: &[T]) -> Vec<NodeId> {
        let first_idx = self.nodes.len();

        for n in ns.iter() {
            let p: Point = n.clone().into();
            self.nodes.push(p);
            self.constraints.push(Constraint::Free);
        }

        let next_idx = self.nodes.len();
        (first_idx..next_idx).map(|x| NodeId(x)).collect()
    }

    pub fn element_descriptors(&self) -> Vec<ElementDescriptor> {
        self.elements
            .iter()
            .map(|e| ElementDescriptor::new(e.node_ids()))
            .collect()
    }

    pub fn add_element(&mut self, desc: ElementDescriptor) {
        let el = build_element(&self, desc);
        // check the element jacobian at the natural coordinate center
        let j_ratio = el.jacobian_ratio();
        // TODO handle this error condition with a result return
        if j_ratio < 0.0 {
            panic!("distorted element: jacobian ratio is {}", j_ratio)
        }
        if j_ratio < 0.033 {
            eprintln!(
                "WARNING: element {} Jacobian is low ({})",
                self.elements.len(),
                j_ratio
            );
        }
        self.elements.push(el);
    }

    pub fn conc_forces(&self) -> Vec<(NodeId, Point)> {
        self.concentrated_forces
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    pub fn add_conc_force(&mut self, n: NodeId, force: Point) {
        self.concentrated_forces.insert(n, force);
    }

    pub fn dist_forces(&self) -> Vec<((NodeId, NodeId), Point)> {
        self.line_forces
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    pub fn add_dist_line_force(&mut self, n: NodeId, m: NodeId, force: Point) {
        self.line_forces.insert((n, m), force);
    }

    pub fn constraints(&self) -> Vec<(NodeId, Constraint)> {
        self.constraints
            .iter()
            .enumerate()
            .map(|(i, &c)| (NodeId(i), c))
            .collect()
    }

    pub fn add_constraint(&mut self, n: NodeId, constraint: Constraint) {
        match (self.constraints[n.0], constraint) {
            (Constraint::Free, c) => {
                self.constraints[n.0] = c;
            }
            (Constraint::AngleFixed(a), Constraint::AngleFixed(b)) => {
                self.constraints[n.0] = Constraint::AngleFixed((a + b) / 2.0);
            }
            (Constraint::AngleFixed(_), c) => {
                self.constraints[n.0] = c;
            }
            (c, Constraint::AngleFixed(_)) => {
                self.constraints[n.0] = c;
            }
            (Constraint::XFixed, Constraint::YFixed) | (Constraint::YFixed, Constraint::XFixed) => {
                self.constraints[n.0] = Constraint::XYFixed;
            }
            _ => {}
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn calc_k(&self) -> Dictionary {
        // evaluate the global K matrix
        let mut res = Dictionary::zeros((self.node_count() * 2, self.node_count() * 2));

        for el in self.elements.iter() {
            for (i, j, coef) in el.calc_k() {
                // i and j are node dofs, coef is the coefficient to add
                let (i_dof, j_dof) = (i.to_dof(), j.to_dof());
                res[(i_dof.0, j_dof.0)] += coef;
            }
        }

        res
    }

    fn calc_m(&self) -> Dictionary {
        // evaluate the global M matrix
        let mut res = Dictionary::zeros((self.node_count() * 2, self.node_count() * 2));

        for el in self.elements.iter() {
            for (i, j, coef) in el.calc_m() {
                // i and j are node dofs, coef is the coefficient to add
                let (i_dof, j_dof) = (i.to_dof(), j.to_dof());
                res[(i_dof.0, j_dof.0)] += coef;
            }
        }

        res
    }

    fn calc_load(&self) -> Vec<f64> {
        // evaluate the global load vector, filling it into the system
        let mut res = vec![0.0; self.node_count() * 2];

        for (&n, &f) in self.concentrated_forces.iter() {
            // check if each node dof still exists and if so apply that part of force
            for d in 0..NODE_DIM {
                let node_dof = NodeDof(n, d);
                res[node_dof.to_dof().0] = f[d];
            }
        }

        for (&(n, m), &f) in self.line_forces.iter() {
            for el in self.elements.iter() {
                // TODO this is a bad way of locating affected elements - very slow, lots of loops
                if el.nodes_connect(n, m) {
                    for (i, val) in el.int_f_l((n, m), f).unwrap() {
                        res[i.to_dof().0] = val;
                    }
                }
            }
        }

        res
    }

    fn raw_to_node_disp(&self, raw_disp: &[f64]) -> Vec<Point> {
        // internal helper to relate raw solution results to deformations
        // reverse masking and transforms of the solver coordinate space
        let mut node_disp = Vec::new();

        let constrainer = DofTransform::from_constraints(&self.constraints);

        let dof_disp = constrainer.unbar_vec(raw_disp);

        for n in self.node_ids() {
            let mut disp = Point::zero(NODE_DIM);
            for d in 0..NODE_DIM {
                let node_dof = NodeDof(n, d);
                disp[d] = dof_disp[node_dof.to_dof().0];
            }
            node_disp.push(disp);
        }

        node_disp
    }

    pub fn calc_displacements<T: Solver>(&mut self) -> Deformation {
        // find the displacements under load and store them in the assemblage
        let constrainer = DofTransform::from_constraints(&self.constraints);

        let k = self.calc_k();
        let r = self.calc_load();

        let k = constrainer.bar_matrix(&k);
        let r = constrainer.bar_vec(&r);

        let mut system = System::from_kr(&k, &LinearMatrix::col_vec(r));

        let solver = T::new(system);

        // TODO properly handle solver errors
        let raw_disp = solver.solve().unwrap();
        let node_disp = self.raw_to_node_disp(raw_disp.as_slice());
        Deformation::new(self, node_disp)
    }

    pub fn calc_modes(&mut self, n: usize) -> Vec<Deformation> {
        // find the n lowest-frequency modes
        // for now, this solution procedure ignores loads
        let constrainer = DofTransform::from_constraints(&self.constraints);

        let k = self.calc_k();
        let k = constrainer.bar_matrix(&k);
        let m = self.calc_m();
        let m = constrainer.bar_matrix(&m);

        let mut sys = System::from_km(&k, &m);

        let eigensys = EigenSystem::new(sys);
        let mut searcher = DeterminantSearcher::new(eigensys);
        let pairs = searcher.find_eigens(n);

        let mut res = Vec::new();

        for p in pairs {
            let mode_shape = self.raw_to_node_disp(p.vector());
            // the solution eigenvalues are the square of the natural frequencies
            let mode_freq = p.value().sqrt();
            res.push(Deformation::with_freq(self, mode_shape, mode_freq));
        }

        res
    }

    pub fn triangles(&self) -> Vec<(usize, usize, usize)> {
        // return all the triangles present in contained elements as triplets of node indices
        let mut res = Vec::new();
        for el in self.elements.iter() {
            res.extend(el.triangles().into_iter().map(|x| (x.0 .0, x.1 .0, x.2 .0)));
        }
        res
    }

    pub fn edges(&self) -> Vec<(usize, usize)> {
        // return all edges present in contained elements as pairs of node inidices
        let mut res = Vec::new();
        for el in self.elements.iter() {
            res.extend(el.edges().into_iter().map(|x| (x.0 .0, x.1 .0)))
        }
        // clear out the duplicates
        res.sort();
        res.dedup();
        res
    }
}

// struct for storing and post-processing displacement results
// elas reference is nice as it ensures Def is always valid
#[derive(Clone, Debug)]
pub struct Deformation<'a> {
    elas: &'a ElementAssemblage,
    node_disp: Vec<Point>,
    ang_freq: Option<f64>,
}

impl<'a> Deformation<'a> {
    fn new(elas: &'a ElementAssemblage, node_disp: Vec<Point>) -> Self {
        Self {
            elas,
            node_disp,
            ang_freq: None,
        }
    }

    fn with_freq(elas: &'a ElementAssemblage, node_disp: Vec<Point>, ang_freq: f64) -> Self {
        Self {
            elas,
            node_disp,
            ang_freq: Some(ang_freq),
        }
    }

    pub fn displacement_norms(&self) -> Vec<f64> {
        self.node_disp.iter().map(|p| p.norm()).collect()
    }

    pub fn displaced_nodes(&self, scale: f64) -> Vec<Point> {
        let mut res = Vec::new();

        for (n, d) in self.elas.nodes.iter().zip(self.node_disp.iter()) {
            res.push(*n + (*d * scale));
        }

        res
    }

    pub fn stresses(&self) -> Vec<Option<stress::StressState>> {
        let mut stress_averages: Vec<Average<StressState>> =
            Vec::with_capacity(self.elas.nodes.len());
        for _ in 0..self.elas.nodes.len() {
            stress_averages.push(Average::new());
        }

        // compute stresses for each node, averaging for shared nodes
        for el in self.elas.elements.iter() {
            let disp_func = Box::new(|nid: NodeId| self.node_disp[nid.0]);
            let el_stresses = el.stresses(disp_func);
            for (nid, s) in el_stresses {
                stress_averages[nid.0].update(s);
            }
        }

        // collect average stresses, setting the stress to None when the average is uninitialized
        stress_averages.into_iter().map(|x| x.consume()).collect()
    }

    pub fn von_mises(&self) -> Vec<f64> {
        // compute the mean von mises yield criterion value for each node, with 0 for unused nodes
        self.stresses()
            .into_iter()
            .map(|x| x.map(|y| y.von_mises()).unwrap_or(0.0))
            .collect()
    }

    // TODO should factor this (quite involved) visual stuff out of element
    pub fn visualize(&self, scale: f64) -> Visualizer {
        let dispn = self.displaced_nodes(scale);

        let mut vis: Visualizer = dispn.into();
        vis.set_edges(self.elas.edges());
        vis.set_triangles(self.elas.triangles());

        vis
    }

    pub fn frequency(&self) -> Option<f64> {
        // associated frequency in rad/sec, if any
        self.ang_freq
    }
}
