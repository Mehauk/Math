use matrix::Matrix;

mod matrix;

fn main() {
    let m = Matrix::from(
        4,
        4,
        vec![
            0.0, 0.0, 3.0, 8.0, 7.0, 9.0, 4.0, 0.0, 6.0, 1.2, 2.2, 3.2, 0.0, 3.4, 0.0, 1.0, 9.1,
        ],
    );
    let mi = Matrix::from(
        4,
        4,
        vec![
            0.0, 0.0, 3.0, 8.0, 7.0, 9.0, 4.0, 0.0, 6.0, 1.2, 2.2, 3.2, 0.0, 3.4, 0.0, 1.0, 9.1,
        ],
    )
    .invert();
    println!("{:?}", m);
    println!("{:?}", mi);
    println!("{:?}", m * mi.unwrap_or(Matrix::identity(4, 4)));
}
