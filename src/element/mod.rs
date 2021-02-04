pub mod material;
pub mod isopar;
pub mod strain;

#[cfg(test)]
mod tests;

pub trait Element {
	fn dims() -> usize;

}