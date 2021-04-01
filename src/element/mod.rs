pub mod integrate;
pub mod isopar;
pub mod loading;
pub mod material;
pub mod strain;
pub mod stress;

use crate::matrix::{Average, Inverse, LinearMatrix, MatrixLike};
use crate::spatial::Point;
use crate::visual::Visualizer;

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
    displacements: Option<Vec<Point>>,
    elements: Vec<IsoparElement>,
    constraints: HashMap<usize, Constraint>,
    concentrated_forces: HashMap<usize, Point>,
    dof_lookup: Option<Vec<[Option<usize>; 3]>>,
    dofs: usize,
    area: Option<f64>,
    thickness: Option<f64>,
    material: Material,
}

impl ElementAssemblage {
    pub fn new(dim: usize, mat: Material) -> Self {
        ElementAssemblage {
            dim,
            nodes: Vec::new(),
            displacements: None,
            elements: Vec::new(),
            constraints: HashMap::new(),
            concentrated_forces: HashMap::new(),
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

    pub fn add_conc_force(&mut self, n: usize, force: Point) {
        self.concentrated_forces.insert(n, force);
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

    fn node_count(&self) -> usize {
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

    fn find_k(&mut self) -> LinearMatrix {
        let mut k = LinearMatrix::zeros(self.dofs);

        for (i, el) in self.elements.iter().enumerate() {
            let int_func = self.k_integrand_func(i);
            let el_k = integrate::nd_gauss_mat(int_func, self.dim, el.integration_order());
            let (el_k_dim, temp) = el_k.shape();
            assert_eq!(el_k_dim, temp);

            println!("element k:\n{}", el_k);
            println!("det element k: {}", el_k.determinant());

            // TODO could probably reduce unnecessary checks here by precomputing
            for i in 0..el_k_dim {
                let (i_node_idx, i_node_dof) = el.i_to_dof(i);
                if let Some(i_dof) = self.find_dof(i_node_idx, i_node_dof) {
                    for j in 0..el_k_dim {
                        let (j_node_idx, j_node_dof) = el.i_to_dof(j);
                        if let Some(j_dof) = self.find_dof(j_node_idx, j_node_dof) {
                            // wooo finally
                            k[(i_dof, j_dof)] += el_k[(i, j)];
                        }
                    }
                }
            }
        }
        println!("{}", k);
        println!("det K: {}", k.determinant());
        k
    }

    pub fn calc_displacements(&mut self) {
        // find the displacements under load and store them in the assemblage
        if self.dof_lookup.is_none() {
            self.compile_lookup();
        }

        let mut assemblage_k = self.find_k();

        let mut con_force = LinearMatrix::zeros((self.dofs, 1));
        // for each node, check if a concentrated force is applied
        for n in 0..self.node_count() {
            if let Some(f) = self.concentrated_forces.get(&n) {
                // check if each node dof still exists and if so apply that part of force
                for d in 0..self.dim {
                    if let Some(dof) = self.find_dof(n, d) {
                        con_force[(dof, 0)] = f[d];
                    }
                }
            }
        }

        // assemble r, the overall load vector
        // for now this is just concentrated loads, but will eventually include body forces etc.
        let r = con_force;

        let raw_displacements = assemblage_k
            .solve_gausselim(r)
            .expect("could not invert stiffness matrix");
        let mut node_displacements = Vec::new();

        for n in 0..self.node_count() {
            let mut disp = Point::zero(self.dim);
            for d in 0..self.dim {
                if let Some(i) = self.find_dof(n, d) {
                    disp[d] = raw_displacements[(i, 0)];
                }
            }
            node_displacements.push(disp);
        }

        self.displacements = Some(node_displacements);
    }

    pub fn displacements(&self) -> Option<Vec<Point>> {
        self.displacements.clone()
    }

    pub fn displacement_norms(&self) -> Option<Vec<f64>> {
        if let Some(disp) = self.displacements() {
            Some(disp.into_iter().map(|d| d.norm()).collect())
        } else {
            None
        }
    }

    pub fn displaced_nodes(&self, scale: f64) -> Option<Vec<Point>> {
        if let Some(disps) = self.displacements.clone() {
            let mut res = Vec::new();
            for (n, d) in self.nodes.iter().zip(disps.into_iter()) {
                res.push(*n + (d * scale));
            }
            Some(res)
        } else {
            None
        }
    }

    pub fn stresses(&self) -> Vec<Option<stress::StressState>> {
        // TODO streamline this, maybe compute disps if necessary
        // TODO take out panics :)

        // set up displacements and stress average structures
        let disps = self
            .displacements
            .as_ref()
            .expect("compute displacements first");

        let mut stress_averages: Vec<Average<StressState>> = Vec::new();
        for _ in 0..self.nodes.len() {
            stress_averages.push(Average::new());
        }

        // compute stresses for each node, averaging for shared nodes
        for el in self.elements.iter() {
            let mut el_u: Vec<f64> = Vec::new();
            for i in 0..el.dofs() {
                let (idx, dof) = el.i_to_dof(i);
                // idx is already an idnex in the global node list
                // dof is the degree of freedom of the node (x/y/z)
                el_u.push(disps[idx][dof]);
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

    // TODO should factor this (quite involved) visual stuff out of element
    pub fn visualize(&self, scale: f64) -> Visualizer {
        let dispn = self
            .displaced_nodes(scale)
            .expect("displacements must first be calculated");

        let mut vis: Visualizer = dispn.into();
        vis.set_edges(self.edges());
        vis.set_triangles(self.triangles());

        vis
    }
}
