use super::MatrixLike;

pub trait Norm {
    fn frobenius(&self) -> f64;
}

impl<T> Norm for T
where
    T: MatrixLike,
{
    fn frobenius(&self) -> f64 {
        // compute the frobenius norm
        let sum: f64 = self.flat().map(|x| x * x).sum();
        sum.sqrt()
    }
}
