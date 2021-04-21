use std::convert::TryInto;

trait Triangulation {
    // the underlying point storage
    type VertexStorage;

    // delaunay-ify the triangulation
    fn delaunay(&mut self);

    // store a new vertex, returning its index
    fn store_vertex<T: TryInto<VertexStorage>>(&mut self, p: T) -> usize;

    // add a triangle to the triangulation, returning true if successful
    fn add_triangle(&mut self, u: usize, v: usize, w: usize) -> bool;

    // delete a triangle, returning true if successful
    fn delete_triangle(&mut self, u: usize, v: usize, w: usize) -> bool;

    // return Some(w) if the positively oriented uvw exists
    fn adjacent(&self, u: usize, v: usize) -> Option<usize>;

    // return an arbitrary triangle including u, if one exists
    fn adjacent_one(&self, u: usize) -> Option<(usize, usize)>;

    // index-based access to underlying geometric predicates

    // determine whether x lies in the oriented triangle tri's circumcircle
    fn in_circle(&self, tri: (usize, usize, usize), x: usize);

    fn bowyer_watson_dig(&mut self, u: usize, v: usize, w: usize) {
        // u is a new vertex
        // check if uvw is delaunay
        if let Some(x) = self.adjacent(w, v) {
            if self.in_circle((u, v, w), x) {
                self.delete_triangle(w, v, x);
                self.bowyer_watson_dig(u, v, x);
                self.bowyer_watson_dig(u, x, w);
            } else {
                self.add_triangle(u, v, w);
            }
        }
    }

    fn bowyer_watson_insert(&mut self, u: usize, tri: (usize, usize, usize)) {
        // insert a vertex into a delaunay triangulation, maintaining the delaunay property
        // tri is a triangle whose cirmcumcircle encloses u
        let (v, w, x) = tri;
        self.delete_triangle(v, w, x);
        self.bowyer_watson_dig(u, v, w);
        self.bowyer_watson_dig(u, w, x);
        self.bowyer_watson_dig(u, x, v);
    }
}
