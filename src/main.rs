mod linear_algebra;
mod machine_learning;

use linear_algebra::matrix;
use machine_learning::load_data;

fn main() -> Result<(), std::io::Error> {
    load_data("src/assets/machine_learning/", "letters")?;

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
