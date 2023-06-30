use std::fmt::Display;

use linear_algebra::Matrix;

fn main() {
    let i = 7;
    let m = Matrix::random(i, i, -10.0, 10.0);
    let mi = m.copy().invert().unwrap();
    println!("{:?}", m);
    println!("{:?}", mi);
    println!("{:?}", m * mi);

    let v = |x, y| {x + y};

    println!("{}", v(1, 2));

}
