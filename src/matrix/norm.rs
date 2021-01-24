use super::Matrix;

impl Matrix {
	pub fn frobenius(&self) -> f64 {
		// compute the frobenius norm
		let sum: f64 = self.data.iter()
			.map(|x| (x.clone(), x.clone()))
			.map(|(x, y)| x * y)
			.sum();
		sum.sqrt()
	}
}