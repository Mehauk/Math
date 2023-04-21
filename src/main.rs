mod matrix;
use matrix::Matrix;

fn main() {
    // let m: Matrix<i32> = Matrix::random(3, 3, -20, 20);
    let m = Matrix {
        m: 3,
        n: 3,
        arr: vec![0, 2, 3, 4, 0, 6, 8, 7, 9],
    };
    m.invert();
    // println!("{:?}", m.invert());
    // let m2 = Matrix::random(2, 3, i32::default(), 5);
    // println!("{:?}", m2);
    // let m3 = m - m2;
    // println!("{:?}", m3);
}
