use crate::matrix::graph::{Graph, Permutation};
use crate::matrix::{Dictionary, MatrixLike};

pub mod direct;
pub mod eigen;
pub mod iterative;

pub trait Solver {
    fn new(sys: System) -> Self;

    // TODO improve matrix::Error API and add to this result
    fn solve(self) -> Result<Vec<f64>, ()>;
}

#[derive(Debug, Clone)]
pub struct System {
    dofs: usize,
    k_mat: Option<Dictionary>,
    m_mat: Option<Dictionary>,
    rhs: Option<Vec<f64>>,
    perm: Option<Permutation>,
}

impl System {
    pub fn new(dofs: usize) -> Self {
        Self {
            dofs,
            k_mat: None,
            m_mat: None,
            rhs: None,
            perm: None,
        }
    }

    pub fn dofs(&self) -> usize {
        self.dofs
    }

    pub fn k_envelope(&self) -> impl Iterator<Item = usize> {
        self.k_mat
            .as_ref()
            .expect("k matrix not initialized")
            .envelope()
            .into_iter()
    }

    pub fn m_envelope(&self) -> impl Iterator<Item = usize> {
        self.m_mat
            .as_ref()
            .expect("m matrix not initialized")
            .envelope()
            .into_iter()
    }

    pub fn add_k_coefficient(&mut self, loc: (usize, usize), val: f64) {
        if let Some(k) = self.k_mat.as_mut() {
            k[loc] += val;
        } else {
            let mut k = Dictionary::zeros(self.dofs);
            k[loc] = val;
            self.k_mat = Some(k);
        }
    }

    pub fn add_m_coefficient(&mut self, loc: (usize, usize), val: f64) {
        if let Some(m) = self.m_mat.as_mut() {
            m[loc] += val;
        } else {
            let mut m = Dictionary::zeros(self.dofs);
            m[loc] = val;
            self.m_mat = Some(m);
        }
    }

    pub fn add_rhs_val(&mut self, loc: usize, val: f64) {
        if let Some(r) = self.rhs.as_mut() {
            r[loc] += val;
        } else {
            let mut r = vec![0.0; self.dofs];
            r[loc] = val;
            self.rhs = Some(r);
        }
    }

    pub fn from_kr<T, U>(k: &T, r: &U) -> Self
    where
        T: MatrixLike,
        U: MatrixLike,
    {
        let dofs = k.shape().0;

        if dofs != k.shape().1 || dofs != r.shape().0 || r.shape().1 != 1 {
            panic!("bad k, r shapes: {:?}, {:?}", k, r);
        }

        let mut res = Self::new(dofs);

        let mut k_mat = Dictionary::zeros(dofs);
        let mut r_vec = vec![0.0; dofs];

        for row in 0..dofs {
            for col in 0..dofs {
                k_mat[(row, col)] = k[(row, col)];
            }
            r_vec[row] = r[(row, 0)];
        }

        res.k_mat = Some(k_mat);
        res.rhs = Some(r_vec);

        res
    }

    pub fn from_km<T, U>(k: &T, m: &U) -> Self
    where
        T: MatrixLike,
        U: MatrixLike,
    {
        let dofs = k.shape().0;
        assert_eq!(k.shape().0, k.shape().1);
        assert_eq!(m.shape().0, m.shape().1);

        let mut res = Self::new(dofs);

        let mut k_mat = Dictionary::zeros(dofs);
        let mut m_mat = Dictionary::zeros(dofs);

        for row in 0..dofs {
            for col in 0..dofs {
                if k[(row, col)] != 0.0 {
                    k_mat[(row, col)] = k[(row, col)];
                }
                if m[(row, col)] != 0.0 {
                    m_mat[(row, col)] = m[(row, col)];
                }
            }
        }

        res.k_mat = Some(k_mat);
        res.m_mat = Some(m_mat);

        res
    }

    pub fn reduce_envelope(&mut self) {
        // apply the reverse cuthill-mkcee ordering, storing the permutation

        if self.perm.is_some() {
            unimplemented!("double permutation");
        }

        // can use edges_lower because k is symmetrical
        let edges = self
            .k_mat
            .as_ref()
            .expect("no k matrix loaded")
            .edges_lower();
        let mut graph = Graph::from_edges(self.dofs(), edges);
        let perm = graph.reverse_cuthill_mckee();

        self.k_mat.as_mut().unwrap().permute(&perm);

        if let Some(m) = self.m_mat.as_mut() {
            m.permute(&perm);
        }

        if let Some(r) = self.rhs.as_mut() {
            perm.permute_slice(r.as_mut_slice());
        }

        self.perm = Some(perm);
    }

    fn into_krp<T, U>(self) -> (T, U, Option<Permutation>)
    where
        T: From<Dictionary>,
        U: From<Vec<f64>>,
    {
        (
            self.k_mat.expect("no k matrix loaded").into(),
            self.rhs.expect("no rhs vector loaded").into(),
            self.perm.map(Permutation::invert),
        )
    }

    fn into_parts<T, U, V>(self) -> (Option<T>, Option<U>, Option<V>, Option<Permutation>)
    where
        T: From<Dictionary>,
        U: From<Dictionary>,
        V: From<Vec<f64>>,
    {
        // return the converted matrices if available
        (
            self.k_mat.map(Into::into),
            self.m_mat.map(Into::into),
            self.rhs.map(Into::into),
            self.perm.map(Permutation::invert),
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn gauss_seidel() {
        use crate::matrix::{CompressedRow, LinearMatrix, MatrixLike, Norm};

        let k = CompressedRow::from_flat(
            4,
            vec![
                5.0, -4.0, 1.0, 0.0, -4.0, 6.0, -4.0, 1.0, 1.0, -4.0, 6.0, -4.0, 0.0, 1.0, -4.0,
                5.0,
            ],
        );
        let r = LinearMatrix::from_flat((4, 1), vec![0.0, 1.0, 0.0, 0.0]);

        let u = super::iterative::gauss_seidel(k, r, 0.0001, 200);

        let target = LinearMatrix::from_flat((4, 1), vec![1.6, 2.6, 2.4, 1.4]);

        assert!((&target - &u).frobenius() <= 0.01);
    }

    #[test]
    fn solver_impls() {
        use super::{
            direct::CholeskyEnvelopeSolver, direct::DenseGaussSolver, iterative::GaussSeidelSolver,
            Solver, System,
        };
        use crate::matrix::{LinearMatrix, MatrixLike};

        let k = LinearMatrix::from_flat(
            4,
            vec![
                5.0, -4.0, 1.0, 0.0, -4.0, 6.0, -4.0, 1.0, 1.0, -4.0, 6.0, -4.0, 0.0, 1.0, -4.0,
                5.0,
            ],
        );
        let r = LinearMatrix::from_flat((4, 1), vec![0.0, 1.0, 0.0, 0.0]);

        let sys = System::from_kr(&k, &r);

        let target = vec![1.6, 2.6, 2.4, 1.4];

        let dg = DenseGaussSolver::new(sys.clone());
        let dg_res = dg.solve().unwrap();
        assert!(
            dg_res
                .iter()
                .zip(target.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                <= 1.0e-8
        );

        let gs = GaussSeidelSolver::new(sys.clone());
        let gs_res = gs.solve().unwrap();
        assert!(
            gs_res
                .iter()
                .zip(target.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                <= 1.0e-2
        );

        let ch = CholeskyEnvelopeSolver::new(sys.clone());
        let ch_res = ch.solve().unwrap();
        assert!(
            ch_res
                .iter()
                .zip(target.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                <= 1.0e-2
        )
    }

    #[test]
    fn envelope_cholesky() {
        use crate::matrix::{Diagonal, LinearMatrix, LowerRowEnvelope, MatrixLike, Norm};

        // testing cholesky decomposition for correctness
        // don't copy this usage pattern for speed-critical operations
        // dense operations and the default mul implementation are slow

        let m = LowerRowEnvelope::from_flat(
            4,
            vec![
                5.0, 0.0, 0.0, 0.0, -4.0, 6.0, 0.0, 0.0, 1.0, -4.0, 6.0, 0.0, 0.0, 1.0, -4.0, 5.0,
            ],
        );

        let l = super::direct::cholesky_envelope(&m);

        let mut l_transpose = LinearMatrix::from_flat(4, l.flat().cloned());
        l_transpose.transpose();

        let m_remade: LinearMatrix = l.mul(&l_transpose);
        let target = LinearMatrix::from_flat(
            4,
            vec![
                5.0, -4.0, 1.0, 0.0, -4.0, 6.0, -4.0, 1.0, 1.0, -4.0, 6.0, -4.0, 0.0, 1.0, -4.0,
                5.0,
            ],
        );

        assert!((&target - &m_remade).frobenius() <= 1e-8);

        let (l, d) = super::direct::cholesky_envelope_no_root(&m);

        let mut l_transpose = LinearMatrix::from_flat(4, l.flat().cloned());
        l_transpose.transpose();

        let temp: LinearMatrix = d.mul(&l_transpose);
        let m_remade: LinearMatrix = l.mul(&temp);

        assert!((&target - &m_remade).frobenius() <= 1e-8);

        // example LDL' decomposition on non positive-definite problem

        let m = LowerRowEnvelope::from_flat(
            4,
            vec![
                2.0, 0.0, 0.0, 0.0, 6.0, 17.0, 0.0, 0.0, 0.0, 2.0, -1.0, 0.0, 6.0, 17.0, -4.0, 33.0,
            ],
        );

        let (l, d) = super::direct::cholesky_envelope_no_root(&m);

        let mut l_transpose = LinearMatrix::from_flat(4, l.flat().cloned());
        l_transpose.transpose();

        let temp: LinearMatrix = d.mul(&l_transpose);
        let m_remade: LinearMatrix = l.mul(&temp);

        let target = LinearMatrix::from_flat(
            4,
            vec![
                2.0, 6.0, 0.0, 6.0, 6.0, 17.0, 2.0, 17.0, 0.0, 2.0, -1.0, -4.0, 6.0, 17.0, -4.0,
                33.0,
            ],
        );

        assert!((&target - &m_remade).frobenius() <= 1e-8);

        let target_diagonal: Diagonal = vec![2.0, -1.0, 3.0, 4.0].into();
        let target_diagonal = LinearMatrix::from_flat(4, target_diagonal.flat().cloned());
        let d = LinearMatrix::from_flat(4, d.flat().cloned());

        assert!((&target_diagonal - &d).frobenius() <= 1e-8);
    }

    #[test]
    fn eigen_determinant_search() {
        use super::eigen::{DeterminantSearcher, EigenSystem};
        use super::System;
        use crate::matrix::{LinearMatrix, MatrixLike};

        let k = LinearMatrix::from_flat(
            4,
            vec![
                5.0, -4.0, 1.0, 0.0, -4.0, 6.0, -4.0, 1.0, 1.0, -4.0, 6.0, -4.0, 0.0, 1.0, -4.0,
                5.0,
            ],
        );

        let m = LinearMatrix::from_flat(
            4,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        );

        let sys = System::from_km(&k, &m);
        let eigsys = EigenSystem::new(sys);
        let mut searcher = DeterminantSearcher::new(eigsys);

        assert_eq!(searcher.eigenvalues_under(2.0), 2);
        assert_eq!(searcher.eigenvalues_under(8.0), 3);

        let pairs = searcher.find_eigens(4);

        println!("{:?}", pairs);

        // check against independently-verified eigenvalues
        assert!((pairs[0].value() - 0.1459).abs() <= 1e-3);
        assert!((pairs[1].value() - 1.9098).abs() <= 1e-3);
        assert!((pairs[2].value() - 6.8541).abs() <= 1e-3);
        assert!((pairs[3].value() - 13.0902).abs() <= 1e-3);

        use crate::matrix::dot;
        assert!(
            (dot(pairs[3].vector(), &[-0.3717, 0.6015, -0.6015, 0.3717]).abs() - 1.0).abs() <= 1e-3
        );
    }
}
