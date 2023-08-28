mod machine_learning;
mod utilities;

use machine_learning::{parse_mnist, DataSet};
use utilities::benchmarking::_time_it;

fn main() -> Result<(), std::io::Error> {
    _matrix_calculations_comparison();
    DataSet::load_data("src/assets/machine_learning/", "letters", parse_mnist)?;

    Ok(())
}

fn _matrix_calculations_comparison() {
    let arr = vec![7.0; 100];

    let nm1 = nalgebra::SMatrix::<f64, 10, 10>::from_vec(arr.clone());
    let nm2 = nalgebra::SMatrix::<f64, 10, 10>::from_vec(arr.clone());

    let nm12 = nalgebra::DMatrix::<f64>::from_vec(10, 10, arr.clone());
    let nm22 = nalgebra::SMatrix::<f64, 10, 10>::from_vec(arr.clone());

    println!("\nMatrix Multiplication:");
    println!("---------");
    println!("Nalgebra Const Const Matrix Multiplication (10, 10) X (10, 10)");
    _time_it(|| nm1 * nm2);
    println!("---");
    println!("Nalgebra Const Dynamic Matrix Multiplication (10, 10) X (10, 10)");
    _time_it(|| nm12 * nm22);
    println!("---------");
}
