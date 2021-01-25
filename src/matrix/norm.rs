use super::MatrixData;

pub trait Norm<T>
    where T: MatrixData
{
    fn frobenius(&self) -> f64;
}

impl<T> Norm<T> for T
    where T: MatrixData
{
    fn frobenius(&self) -> f64 {
        // compute the frobenius norm
        let sum: f64 = self
            .data
            .iter()
            .map(|x| (x.clone(), x.clone()))
            .map(|(x, y)| x * y)
            .sum();
        sum.sqrt()
    }
}
