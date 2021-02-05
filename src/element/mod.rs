pub mod material;
pub mod isopar;
pub mod strain;
pub mod integrate;

#[cfg(test)]
mod tests;

pub trait Element {
	fn dims() -> usize;

}