pub mod material;
pub mod isopar;

#[cfg(test)]
mod tests;

pub trait Element {
	fn dims() -> usize;

}