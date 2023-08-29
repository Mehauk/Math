use std::time::SystemTime;

pub fn _time_it<T>(func: impl FnOnce() -> T) {
    let s = SystemTime::now();
    func();
    let e = SystemTime::now();
    let duration = e.duration_since(s).unwrap();
    println!("Time Elapsed: {} seconds", duration.as_secs_f64())
}

fn _matrix_calculations_comparison() {
    let arr = vec![7.0; 100];
    let arr2 = vec![7.0; 10000];

    let nmr = nalgebra::SMatrix::<f64, 100, 1>::from_vec(arr.clone());
    let nmc = nalgebra::SMatrix::<f64, 1, 100>::from_vec(arr.clone());

    let nmsq = nalgebra::SMatrix::<f64, 100, 100>::from_vec(arr2.clone());

    println!("---------");
    println!("---------");
    println!("---------");

    println!("\nNalgebra row vs column multiplication:");
    println!("---------");
    println!("Matrix Multiplication (100, 100) X (100, 1)");
    _time_it(|| nmsq * nmr);
    println!("---");
    println!("Matrix Multiplication (1, 100) X (100, 100)");
    _time_it(|| nmc * nmsq);
    println!("---------");
}
