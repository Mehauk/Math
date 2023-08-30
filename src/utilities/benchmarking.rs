use std::time::SystemTime;

pub fn _time_it<T>(func: impl FnOnce() -> T) {
    let s = SystemTime::now();
    func();
    let e = SystemTime::now();
    let duration = e.duration_since(s).unwrap();
    println!("Time Elapsed: {} seconds", duration.as_secs_f64())
}

pub fn _matrix_calculations_comparison() {
    let arr = vec![7.0; 28 * 28];
    let arr2 = vec![7.0; 28 * 28 * 10];

    let nmc = nalgebra::SMatrix::<f64, { 28 * 28 }, 1>::from_vec(arr.clone());

    let nmsq = nalgebra::SMatrix::<f64, 10, { 28 * 28 }>::from_vec(arr2.clone());

    println!("---------");
    println!("---------");
    println!("---------");

    println!("\nNalgebra row vs column multiplication:");
    println!("---------");
    println!("Matrix Multiplication (10, 28^2) X (28^2, 1)");
    _time_it(|| nmsq * nmc);
    println!("---");
    println!("---------");
}
