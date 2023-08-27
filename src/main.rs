mod linear_algebra;
mod machine_learning;
mod utilities;

use linear_algebra::matrix;
use machine_learning::{parse_mnist, DataSet};
use utilities::benchmarking::_time_it;

fn main() -> Result<(), std::io::Error> {
    // _matrix_calculations_comparison();
    DataSet::load_data("src/assets/machine_learning/", "letters", parse_mnist)?;

    Ok(())
}

fn _matrix_calculations_comparison() {
    let arr = vec![7.0; 100];
    let arr2 = vec![7.0; 10000];
    let m1 = matrix::CustomMatrix::from(10, 10, arr.clone());
    let m2 = matrix::CustomMatrix::from(10, 10, arr.clone());

    let nm1 = nalgebra::SMatrix::<f64, 10, 10>::from_vec(arr.clone());
    let nm2 = nalgebra::SMatrix::<f64, 10, 10>::from_vec(arr.clone());

    let nmr = nalgebra::SMatrix::<f64, 100, 1>::from_vec(arr.clone());
    let nmc = nalgebra::SMatrix::<f64, 1, 100>::from_vec(arr.clone());

    let nmsq = nalgebra::SMatrix::<f64, 100, 100>::from_vec(arr2.clone());

    println!("\nMatrix Multiplication:");
    println!("---------");
    println!("Custom Matrix Multiplication (10, 10) X (10, 10)");
    _time_it(|| m1 * m2);
    println!("---");
    println!("Nalgebra Matrix Multiplication (10, 10) X (10, 10)");
    _time_it(|| nm1 * nm2);
    println!("---------");

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
