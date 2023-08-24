mod linear_algebra;
mod machine_learning;

use std::fs::File;

use linear_algebra::matrix;

fn main() -> Result<(), std::io::Error> {
    let s = std::fs::read("src/assets/r.txt")?;

    println!("{:?}", s);

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
