use std::collections::HashSet;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct Graph {
    adjacent: Vec<usize>,
    indices: Vec<usize>,
    temp: Vec<usize>,
}

impl Graph {
    pub fn from_lol<T, U>(adjs: T) -> Self
    where
        T: IntoIterator<Item = U>,
        U: IntoIterator<Item = usize>,
    {
        // create an ordered graph from a list of lists,
        // where adj[i][_] = j means {i, j} is an element of E
        let mut res = Self {
            adjacent: Vec::new(),
            indices: Vec::new(),
            temp: Vec::new(),
        };

        for adj in adjs.into_iter() {
            res.indices.push(res.adjacent.len());
            for a in adj.into_iter() {
                res.adjacent.push(a);
            }
        }

        res
    }

    pub fn from_edges<T>(v_count: usize, edge_iter: T) -> Self
    where 
        T: IntoIterator<Item = (usize, usize)>
    {
        // construct a graph from directed edges

        // ensure edge list is clean and add reverses
        let mut edges = HashSet::new();
        for (x, y) in edge_iter.into_iter() {
            if x >= v_count || y >= v_count {
                panic!("edge vertex out of bounds");
            }

            if x == y { continue }

            edges.insert((x, y));
            edges.insert((y, x));
        }

        let mut edges: Vec<(usize, usize)> = edges.into_iter().collect();

        // sort edges by first vertex then second edge (ascending)
        edges.sort_by(|x, y| match x.0.cmp(&y.0) {
            Ordering::Equal => x.1.cmp(&y.1),
            o => o,
        });

        let mut res = Self {
            adjacent: Vec::new(),
            indices: Vec::new(),
            temp: Vec::new(),
        };

        for (x, y) in edges.into_iter() {
            while (x + 1) > res.indices.len() {
                res.indices.push(res.adjacent.len());
            }

            res.adjacent.push(y);
        }

        res
    }

    pub fn vertex_count(&self) -> usize {
        self.indices.len()
    }

    fn adj_idxs(&self, v: usize) -> (usize, usize) {
        // helper function to locate adjacencies in data
        let start = self.indices[v];
        let end = self
            .indices
            .get(v + 1)
            .cloned()
            .unwrap_or(self.adjacent.len());
        (start, end)
    }

    pub fn adjacent(&self, v: usize) -> &[usize] {
        if v >= self.vertex_count() {
            panic!("vertex index out of bounds")
        }

        let (start, end) = self.adj_idxs(v);
        &self.adjacent[start..end]
    }

    pub fn adjacent_masked<'a>(&'a mut self, i: usize, mask: &Vec<bool>) -> &'a [usize] {
        if i >= self.vertex_count() {
            panic!("vertex index out of bounds")
        }

        let (start, end) = self.adj_idxs(i);

        self.temp.clear();
        for x in start..end {
            let v = self.adjacent[x];
            if mask[v] {
                self.temp.push(v);
            }
        }

        self.temp.as_slice()
    }

    pub fn degree(&self, v: usize) -> usize {
        // degree of node n

        let (start, end) = self.adj_idxs(v);
        end - start
    }

    pub fn degree_masked(&self, v: usize, mask: &Vec<bool>) -> usize {
        let (start, end) = self.adj_idxs(v);

        (start..end)
            .map(|i| if mask[self.adjacent[i]] { 1 } else { 0 })
            .sum()
    }

    pub fn far_node(&mut self, root: usize) -> (usize, usize) {
        // find the node furthest from root on the graph and the distance to it
        // returns (node, length)
        // ties are broken by smaller degree

        if root >= self.vertex_count() {
            panic!("root label out of bounds");
        }

        let mut cur_level = vec![root];
        let mut next_level = Vec::new();
        let mut mask = vec![true; self.vertex_count()];
        let mut length = 0;

        mask[root] = false;

        loop {
            next_level.clear();

            // explore the unvisited nodes adjacent to every node on the current level
            for v in cur_level.iter().cloned() {
                mask[v] = false;
                let adj = self.adjacent_masked(v, &mask);
                for a in adj.iter().cloned() {
                    mask[a] = false;
                    next_level.push(a);
                }
            }

            if next_level.is_empty() {
                break;
            } else {
                // next level becomes the current level
                // (this way of swapping should prevent reallocations)
                length += 1;
                std::mem::swap(&mut cur_level, &mut next_level);
            }
        }

        // cur_level contains the last level to be explored
        // select the element with the smallest degree

        let res_node = cur_level
            .into_iter()
            .min_by(|&v, &w| self.degree(v).cmp(&self.degree(w)))
            .unwrap();
        (res_node, length)
    }

    pub fn pseudo_peripheral(&mut self) -> usize {
        // identify a pseudoperipheral node of the graph by George and Liu's method

        // choose an arbitrary starting node
        let mut x = 0;
        let mut longest = 0;

        loop {
            let (new_x, length) = self.far_node(x);
            x = new_x;

            if length > longest {
                longest = length;
            } else {
                break x
            }
        }
    }

    pub fn reverse_cuthill_mckee(&mut self) -> Permutation {
        // find the reverse cuthill mckee labelling of the graph's nodes

        let mut ordering = Vec::new();
        ordering.push(self.pseudo_peripheral());

        let mut mask = vec![true; self.vertex_count()];
        mask[ordering[0]] = false;

        for i in 0..self.vertex_count() {
            let adj = self.adjacent_masked(ordering[i], &mask);

            for a in adj {
                mask[*a] = false;
                ordering.push(*a);
            }
        }

        assert_eq!(ordering.len(), self.vertex_count());
        ordering.reverse();
        Permutation::from_placement(ordering)
    }
}

impl PartialEq for Graph {
    fn eq(&self, other: &Self) -> bool {
        self.adjacent == other.adjacent && self.indices == other.indices
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Permutation {
    forward: Vec<usize>,
    back: Vec<usize>,
}

impl Permutation {
    pub(super) fn from_placement(p: Vec<usize>) -> Self {
        // build a permutation from a placement
        // where value n being in the ith place means n transforms to i

        let len = p.len();
        let mut res = Self { forward: vec![0; len], back: vec![0; len], };

        // confirm values are unique and bounded properly
        let mut items = HashSet::new();
        assert!(p.iter().all(move |x| items.insert(x) && *x < len), "improper permutation");

        for (i, x) in p.into_iter().enumerate() {
            res.forward[x] = i;
            res.back[i] = x;
        }

        res
    }

    pub fn permute(&self, x: usize) -> usize {
        self.forward[x]
    }

    pub fn unpermute(&self, x: usize) -> usize {
        self.back[x]
    }
}