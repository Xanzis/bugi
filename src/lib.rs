pub mod element;
pub mod file;
mod matrix;
pub mod mesher;
mod spatial;
pub mod visual;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
