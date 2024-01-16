// use std::{ops::Index, vec};

// macro_rules! index_impl {
//     ([$($i:ty),*], $t:ty, $o:ty) => {

//         impl Index<($($i,)*)> for $t {
//             type Output = $o;

//             fn index(&self, index: ($($i,)*)) -> &$o {

//             }

//         }
//         forward_ref_index! {}
//     };
// }

// macro_rules! forward_ref_index {
//     () => {};
// }

// struct Container {
//     values: Vec<f64>,
//     values1: Vec<f64>,
// }

// fn main() {
//     let c = Container {
//         values: vec![1.0, 2.0, 3.0],
//         values1: vec![1.0, 2.0, 3.0],
//     };
//     index_impl!([usize, usize], Container, f64);
//     println!("{}", c[(1, 1)])
// }

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn test_saving_and_loading() {}
// }
