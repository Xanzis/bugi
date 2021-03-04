pub mod material;
pub mod isopar;
pub mod strain;
pub mod integrate;
mod loading;

use crate::spatial::Point;
use crate::matrix::{LinearMatrix, MatrixLike, Inverse};

use strain::StrainRule;
use loading::{Constraint};
use isopar::{IsoparElement, Bar2Node, PlaneNNode};
use material::{ProblemType, AL6061, TEST};

use std::collections::{HashMap};

#[cfg(test)]
mod tests;

// type containing element variety and node index information
// TODO this is a silly indirection and hides some underlying APIs
// once things are working, factor this out
// TODO make this non-public once changes are applied
#[derive(Debug, Clone)]
pub enum ElementMap {
	IsoB2N(usize, usize),
	IsoPNN(Vec<usize>),
}

// the big one - primary puclic-facing API for this crate
#[derive(Debug, Clone)]
pub struct ElementAssemblage {
	dim: usize,
	nodes: Vec<Point>,
	displacements: Option<Vec<Point>>,
	elements: Vec<ElementMap>,
	constraints: HashMap<usize, Constraint>,
	concentrated_forces: HashMap<usize, Point>,
	dof_lookup: Option<Vec<[Option<usize>; 3]>>,
	dofs: usize,
}

impl ElementAssemblage {
	pub fn new(dim: usize) -> Self {
		ElementAssemblage {
			dim,
			nodes: Vec::new(),
			displacements: None,
			elements: Vec::new(),
			constraints: HashMap::new(),
			concentrated_forces: HashMap::new(),
			dof_lookup: None,
			dofs: 0,
		}
	}

	pub fn nodes(&self) -> Vec<Point> {
		self.nodes.clone()
	}

	// TODO streamline adding nodes and associated elements
	pub fn add_nodes<T: Into<Point>>(&mut self, ns: Vec<T>) {
		for n in ns.into_iter() {
			let p: Point = n.into();
			if p.dim() != self.dim { panic!("bad node dimension") }
			self.nodes.push(p);
		}
	}

	pub fn add_element(&mut self, el: ElementMap) {
		self.elements.push(el);
	}

	pub fn add_conc_force(&mut self, n: usize, force: Point) {
		self.concentrated_forces.insert(n, force);
	}

	pub fn add_constraint(&mut self, n: usize, constraint: Constraint) {
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
						if constraint.plain_dim_struck(d) { continue; }
						node_lookup[d] = Some(i);
						i += 1;
					}
					lookup.push(node_lookup);
				}
				else {
					unimplemented!()
				}
			}
			else {
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

	fn k_integrand_func(&self, i: usize) -> (Vec<usize>, Box<dyn Fn(Point) -> LinearMatrix>) {
		// get a closure for computing an elements' K integrand at a given location
		// also return a vector of the **points** used by the element, in order (constraint-agnostic)
		// TODO add assemblage-wide material specification
		match self.elements[i].clone() {
			ElementMap::IsoB2N(a, b) => {
				let el = Bar2Node::new(vec![self.nodes[a], self.nodes[b]]);
				let strain_rule = StrainRule::Bar;
				let c = AL6061.get_c(ProblemType::Bar);
				// TODO el should probably internally cache some intermediate matrices
				// right now it recomputes the interpolations each time
				let k_func = move |p| el.find_k_integrand(p, &c, strain_rule);
				(vec![a, b], Box::new(k_func))
			},
			ElementMap::IsoPNN(a) => {
				let points: Vec<Point> = a.clone().into_iter().map(|x| self.nodes[x]).collect();
				let el = PlaneNNode::new(points);
				let strain_rule = StrainRule::PlaneStrain;
				let c = AL6061.get_c(ProblemType::PlaneStress);
				//let c = TEST.get_c(ProblemType::PlaneStress);
				let k_func = move |p| el.find_k_integrand(p, &c, strain_rule);
				(a, Box::new(k_func))
			}
		}
	}

	fn find_k(&mut self) -> LinearMatrix {
		let mut k = LinearMatrix::zeros(self.dofs);

		for i in 0..self.elements.len() {
			let (el_nodes, int_func) = self.k_integrand_func(i);
			// TODO choose integration order more cleverly
			// underlying element should provide an integration order
			let el_k = integrate::nd_gauss_mat(int_func, self.dim, 2);
			let (el_k_dim, temp) = el_k.shape();
			assert_eq!(el_k_dim, temp);

			println!("element k:\n{}", el_k);
			println!("det element k: {}", el_k.determinant());

			// TODO this is bad indirection, elements should be holding actual IsoParElements
			// (they already have this function)
			let el_node_count = match &self.elements[i] {
				ElementMap::IsoB2N(_, _) => 2,
				ElementMap::IsoPNN(a) => a.len(),
			};

			// TODO this is gross, also generalize for dimensionality differences
			// TODO underlying element should have function for relating one of its dofs
			//   to a node index (in the element frame) and a node dof, for use in find_dof
			// TODO could also reduce unnecessary checks by pre-computing a hashmap
			for i in 0..el_k_dim {
				// TODO following line is part to change
				let i_node_idx = el_nodes[i / self.dim];
				let i_node_dof = i % self.dim;
				if let Some(i_dof) = self.find_dof(i_node_idx, i_node_dof) {
					for j in 0..el_k_dim {
						let j_node_idx = el_nodes[j / self.dim];
						let j_node_dof = j % self.dim;
						if let Some(j_dof) = self.find_dof(j_node_idx, j_node_dof) {
							// wooo finally
							k[(i_dof, j_dof)] = el_k[(i, j)];
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
		if self.dof_lookup.is_none() { self.compile_lookup(); }

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

		let raw_displacements = assemblage_k.solve_gausselim(r).expect("could not invert stiffness matrix");
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

	pub fn displaced_nodes(&self, scale: f64) -> Option<Vec<Point>> {
		if let Some(disps) = self.displacements.clone() {
			let mut res = Vec::new();
			for (n, d) in self.nodes.iter().zip(disps.into_iter()) {
				res.push(*n + (d * scale));
			}
			Some(res)
		}
		else { None }
	}
}