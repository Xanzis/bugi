use crate::matrix::{Matrix, LinearMatrix};
use crate::spatial::Point;

pub mod material;

pub trait Element {
	fn dims() -> usize;

}