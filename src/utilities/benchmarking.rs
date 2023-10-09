use std::time::SystemTime;

use crate::calculus::functions::_sigmoid_derivative;

pub fn _time_it<T>(func: impl FnOnce() -> T) {
    let s = SystemTime::now();
    func();
    let e = SystemTime::now();
    let duration = e.duration_since(s).unwrap();
    println!("Time Elapsed: {} seconds", duration.as_secs_f64())
}

pub fn _matrix_calculations_comparison() {
    // let arr = vec![7.0; 28 * 28];
    let arr2 = vec![7.0; 28 * 28 * 10];
    let arr3 = vec![1.0; 28 * 28 * 10];

    // let nmc = nalgebra::SMatrix::<f64, { 28 * 28 }, 1>::from_vec(arr.clone());

    let mut nmsq = nalgebra::SMatrix::<f64, 10, { 28 * 28 }>::from_vec(arr2.clone());
    let mut nmsq2 = nalgebra::SMatrix::<f64, 10, { 28 * 28 }>::from_vec(arr2.clone());
    let ones = nalgebra::SMatrix::<f64, 10, { 28 * 28 }>::from_vec(arr3.clone());

    println!("---------");
    println!("---------");
    println!("---------");

    println!("\nNalgebra element wise manipulation:");
    println!("---------");
    println!("Matrix sigmoid derivative element wise");
    _time_it(|| nmsq.apply(_sigmoid_derivative));
    println!("---");
    println!("Matrix sigmoid derivative by Matrix Arithmetic");
    _time_it(|| nmsq2.component_mul_assign(&(ones - nmsq2)));
    println!("---");
    println!("---------");
}
