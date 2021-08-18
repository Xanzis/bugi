pub mod integrate;
pub mod isopar;
pub mod loading;
pub mod material;
pub mod strain;
pub mod stress;

use crate::matrix::solve::eigen::{DeterminantSearcher, EigenSystem};
use crate::matrix::solve::{Solver, System};
use crate::matrix::{Average, LinearMatrix, MatrixLike};
use crate::spatial::Point;
use crate::visual::Visualizer;

use isopar::ElementType;
use isopar::IsoparElement;
use loading::Constraint;
use material::Material;
use stress::StressState;

use std::collections::{HashMap, HashSet};

#[cfg(test)]
mod tests;

// the big one - primary puclic-facing API for this crate
#[derive(Debug, Clone)]
pub struct ElementAssemblage {
    dim: usize,
    nodes: Vec<Point>,
    dofs: usize,

    area: Option<f64>,
    thickness: Option<f64>,
    material: Material,

    elements: Vec<IsoparElement>,
    constraints: HashMap<usize, Constraint>,
    concentrated_forces: HashMap<usize, Point>,
    line_forces: HashMap<(usize, usize), Point>,

    dof_lookup: Option<Vec<[Option<usize>; 3]>>,
}

impl ElementAssemblage {
    pub fn new(dim: usize, mat: Material) -> Self {
        ElementAssemblage {
            dim,
            nodes: Vec::new(),
            elements: Vec::new(),
            constraints: HashMap::new(),
            concentrated_forces: HashMap::new(),
            line_forces: HashMap::new(),
            dof_lookup: None,
            dofs: 0,
            area: None,
            thickness: None,
            material: mat,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn thickness(&self) -> Option<f64> {
        self.thickness
    }

    pub fn set_thickness(&mut self, thickness: f64) {
        self.thickness = Some(thickness);
    }

    pub fn area(&self) -> Option<f64> {
        self.area
    }

    pub fn set_area(&mut self, area: f64) {
        self.area = Some(area)
    }

    pub fn material(&self) -> Material {
        self.material
    }

    pub fn node(&self, n: usize) -> Point {
        // TODO maybe make a non-panicking version
        self.nodes[n]
    }

    pub fn nodes(&self) -> Vec<Point> {
        self.nodes.clone()
    }

    pub fn add_nodes<T: Into<Point>>(&mut self, ns: Vec<T>) {
        for n in ns.into_iter() {
            let p: Point = n.into();
            if p.dim() != self.dim {
                panic!("bad node dimension")
            }
            self.nodes.push(p);
        }
    }

    pub fn element_node_idxs(&self) -> Vec<Vec<usize>> {
        self.elements.iter().map(|e| e.node_idxs()).collect()
    }

    pub fn element_types(&self) -> Vec<ElementType> {
        self.elements.iter().map(|e| e.el_type()).collect()
    }

    pub fn add_element(&mut self, node_idxs: Vec<usize>) {
        // el should be an El::blank(), it'll get overwritten
        let el = IsoparElement::new(&self.nodes, node_idxs, self.material());
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

    // TODO add streamlined add_element alternative for already-created elements
    // (to be more efficient for elements where the node ordering is already known)
    // also allow specification of element subtypes (eg triangular/rectangular)
    // for when it is difficult to infer (six-node triangle or 6-node rectangle?)

    pub fn conc_forces(&self) -> Vec<(usize, Point)> {
        self.concentrated_forces
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    pub fn add_conc_force(&mut self, n: usize, force: Point) {
        self.concentrated_forces.insert(n, force);
    }

    pub fn dist_forces(&self) -> Vec<((usize, usize), Point)> {
        self.line_forces
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    pub fn add_dist_line_force(&mut self, n: usize, m: usize, force: Point) {
        self.line_forces.insert((n, m), force);
    }

    pub fn constraints(&self) -> Vec<(usize, Constraint)> {
        self.constraints
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    pub fn add_constraint(&mut self, n: usize, constraint: Constraint) {
        // TODO avoid overwriting existing constraints
        self.constraints.insert(n, constraint);
    }

    fn find_dof(&self, node_idx: usize, dim_idx: usize) -> Option<usize> {
        // find the index of a node's dof in the global dof lookup
        // return None if the lookup is empty or the dof has been cancelled by a constraint
        match &self.dof_lookup {
            None => None,
            Some(l) => l[node_idx][dim_idx],
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn compile_lookup(&mut self) {
        // assign dof indices to dof_lookup and count the dofs
        // TODO: add handling for constraints in transformed coordinate systems
        let mut i = 0;
        let mut lookup = Vec::new();

        for p in 0..self.nodes.len() {
            if let Some(constraint) = self.constraints.get(&p) {
                // there is a constraint on node p
                if constraint.is_plain() {
                    let mut node_lookup = [None, None, None];
                    for d in 0..self.dim {
                        if constraint.plain_dim_struck(d) {
                            continue;
                        }
                        node_lookup[d] = Some(i);
                        i += 1;
                    }
                    lookup.push(node_lookup);
                } else {
                    unimplemented!()
                }
            } else {
                // node p is unconstrained
                let mut node_lookup = [None, None, None];
                for d in 0..self.dim {
                    node_lookup[d] = Some(i);
                    i += 1;
                }
                lookup.push(node_lookup);
            }
        }

        self.dof_lookup = Some(lookup);
        self.dofs = i;
    }

    fn k_integrand_func(&self, i: usize) -> Box<dyn Fn(Point) -> LinearMatrix> {
        // get a closure for computing an elements' K integrand at a given location
        // (wraps the underlying integrand method, with a multiplier for area/thickness of element)
        // TODO make plane strain/axisym possible to select (rolling with plane stress for now)

        let multiplier = match self.dim {
            1 => self.area().expect("missing bar/beam area"),
            2 => self.thickness().expect("missing plane thickness"),
            3 => 1.0,
            _ => unimplemented!(),
        };

        let el = self
            .elements
            .get(i)
            .expect("out of bounds element id")
            .clone();

        let k_func = move |p| {
            let mut k = el.find_k_integrand(p);
            k *= multiplier;
            k
        };

        Box::new(k_func)
    }

    fn m_integrand_func(&self, i: usize) -> Box<dyn Fn(Point) -> LinearMatrix> {
        // get a colosure for computing the M integrand at a given location

        let multiplier = match self.dim {
            1 => self.area().expect("missing bar/beam area"),
            2 => self.thickness().expect("missing plane thickness"),
            3 => 1.0,
            _ => unimplemented!(),
        };

        let el = self
            .elements
            .get(i)
            .expect("element id out of bounds")
            .clone();

        let m_func = move |p| {
            let mut m = el.find_m_integrand(p);
            m *= multiplier;
            m
        };

        Box::new(m_func)
    }

    fn calc_k(&self, sys: &mut System) {
        // evaluate the global K matrix, filling it into sys

        for (i, el) in self.elements.iter().enumerate() {
            let int_func = self.k_integrand_func(i);
            let el_k = integrate::nd_gauss_mat(int_func, self.dim, el.integration_order());
            let (el_k_dim, temp) = el_k.shape();
            assert_eq!(el_k_dim, temp);

            // TODO could probably reduce unnecessary checks here by precomputing
            for i in 0..el_k_dim {
                let (i_node_idx, i_node_dof) = el.i_to_dof(i);
                if let Some(i_dof) = self.find_dof(i_node_idx, i_node_dof) {
                    for j in 0..el_k_dim {
                        let (j_node_idx, j_node_dof) = el.i_to_dof(j);
                        if let Some(j_dof) = self.find_dof(j_node_idx, j_node_dof) {
                            // wooo finally
                            sys.add_k_coefficient((i_dof, j_dof), el_k[(i, j)]);
                        }
                    }
                }
            }
        }
    }

    fn calc_m(&self, sys: &mut System) {
        // evaluate the global M matrix

        for (i, el) in self.elements.iter().enumerate() {
            let int_func = self.m_integrand_func(i);
            let el_m = integrate::nd_gauss_mat(int_func, self.dim, el.integration_order());
            let (el_m_dim, temp) = el_m.shape();
            assert_eq!(el_m_dim, temp);

            // TODO could probably reduce unnecessary checks here by precomputing
            for i in 0..el_m_dim {
                let (i_node_idx, i_node_dof) = el.i_to_dof(i);
                if let Some(i_dof) = self.find_dof(i_node_idx, i_node_dof) {
                    for j in 0..el_m_dim {
                        let (j_node_idx, j_node_dof) = el.i_to_dof(j);
                        if let Some(j_dof) = self.find_dof(j_node_idx, j_node_dof) {
                            // wooo finally
                            sys.add_m_coefficient((i_dof, j_dof), el_m[(i, j)]);
                        }
                    }
                }
            }
        }
    }

    fn calc_load(&self, sys: &mut System) {
        // evaluate the global load vector, filling it into the system

        for (&n, &f) in self.concentrated_forces.iter() {
            // check if each node dof still exists and if so apply that part of force
            for d in 0..self.dim() {
                if let Some(dof) = self.find_dof(n, d) {
                    sys.add_rhs_val(dof, f[d]);
                }
            }
        }

        for (&(n, m), &f) in self.line_forces.iter() {
            // convert force f to a column vector
            let f = LinearMatrix::col_vec(f.to_vec());
            for el in self.elements.iter() {
                // TODO this is a bad way of locating affected elements - very slow, lots of loops
                if el.nodes_connect(n, m) {
                    let f_l = el.find_f_l(n, m, &f);

                    for i in 0..el.dofs() {
                        // for each value in the interpolated / integrated force vector,
                        // find the corresponding degree of freedom
                        let (node_idx, node_dof) = el.i_to_dof(i);
                        // check if the degree of freedom still exists, and if so add the force
                        if let Some(dof) = self.find_dof(node_idx, node_dof) {
                            // TODO have f_l be a full Matrix seems silly
                            sys.add_rhs_val(dof, f_l[(i, 0)])
                        }
                    }
                }
            }
        }

        // TODO add area body forces / initial loads
    }

    fn raw_to_node_disp(&self, raw_disp: &[f64]) -> Vec<Point> {
        // internal helper to relate raw solution results to deformations
        // use the dof lookup to fill out node displacements
        assert_eq!(raw_disp.len(), self.dofs);

        let mut node_disp = Vec::new();

        for n in 0..self.node_count() {
            let mut disp = Point::zero(self.dim);
            for d in 0..self.dim {
                if let Some(i) = self.find_dof(n, d) {
                    disp[d] = raw_disp[i];
                }
            }
            node_disp.push(disp);
        }

        node_disp
    }

    pub fn calc_displacements<T: Solver>(&mut self) -> Deformation {
        // find the displacements under load and store them in the assemblage
        if self.dof_lookup.is_none() {
            self.compile_lookup();
        }

        let mut system = System::new(self.dofs);
        self.calc_k(&mut system);
        self.calc_load(&mut system);

        let solver = T::new(system);

        // TODO properly handle solver errors
        let raw_disp = solver.solve().unwrap();

        Deformation::new(self, self.raw_to_node_disp(raw_disp.as_slice()))
    }

    pub fn calc_modes(&mut self, n: usize) -> Vec<Deformation> {
        // find the n lowest-frequency modes
        // for now, this solution  procedure ignored loads
        if self.dof_lookup.is_none() {
            self.compile_lookup();
        }

        let mut sys = System::new(self.dofs);
        self.calc_k(&mut sys);
        self.calc_m(&mut sys);

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
            if let Some(tris) = el.triangles() {
                res.extend(tris);
            }
        }
        res
    }

    pub fn edges(&self) -> Vec<(usize, usize)> {
        // return all edges present in contained elements as pairs of node inidices
        let mut res = Vec::new();
        for el in self.elements.iter() {
            if let Some(edgs) = el.edges() {
                res.extend(edgs);
            }
        }
        // clear out the duplicates
        let res: HashSet<(usize, usize)> = res.into_iter().collect();
        res.into_iter().collect()
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
        let mut stress_averages: Vec<Average<StressState>> = Vec::new();
        for _ in 0..self.elas.nodes.len() {
            stress_averages.push(Average::new());
        }

        // compute stresses for each node, averaging for shared nodes
        for el in self.elas.elements.iter() {
            let mut el_u: Vec<f64> = Vec::new();
            for i in 0..el.dofs() {
                let (idx, dof) = el.i_to_dof(i);
                // idx is already an index in the global node list
                // dof is the degree of freedom of the node (x/y/z)
                el_u.push(self.node_disp[idx][dof]);
            }
            for i in 0..el.node_count() {
                let nstress = el.node_stress(i, el_u.clone());
                stress_averages[el.node_idx(i)].update(nstress);
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
}
