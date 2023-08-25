mod linear_algebra;
mod machine_learning;

use linear_algebra::matrix;
use machine_learning::{DataSet, MnistDatasetParser};

fn main() -> Result<(), std::io::Error> {
    DataSet::load_data(
        "src/assets/machine_learning/",
        "letters",
        &MnistDatasetParser {},
    )?;

    Ok(())
}

fn test_custom_matrix_methods() {
    let i = 7;
    let m = matrix::CustomMatrix::random(i, i, -10.0, 10.0);
    let mi = m.copy().invert().unwrap();
    println!("{:?}", m);
    println!("{:?}", mi);
    println!("{:?}", m.copy() * mi);
    println!("{:?}", m.transpose());
}
