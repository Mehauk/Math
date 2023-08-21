mod linear_algebra;

use linear_algebra::matrix;

fn main() {
    let i = 7;
    let m = matrix::CustomMatrix::random(i, i, -10.0, 10.0);
    let mi = m.copy().invert().unwrap();
    println!("{:?}", m);
    println!("{:?}", mi);
    println!("{:?}", m.copy() * mi);
    println!("{:?}", m.transpose());
}
