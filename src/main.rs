use num_traits::{Num, One};
use rand::{
    distributions::{uniform::SampleUniform, Uniform},
    Rng,
};
use std::{
    fmt::{Debug, Display},
    iter::Sum,
    ops::{Add, AddAssign, Mul, Sub, SubAssign},
};

struct Matrix<T> {
    m: usize,
    n: usize,
    arr: Vec<T>,
}

impl<T: Num + Copy + Clone + SampleUniform + Default + One + Display + AddAssign> Matrix<T> {
    fn new(m: usize, n: usize) -> Self {
        let v = vec![T::default(); n * m];
        Matrix { m: m, n: n, arr: v }
    }

    fn random(m: usize, n: usize, min: T, max: T) -> Self {
        let mut v: Vec<T> = vec![];
        let range = Uniform::new(min, max);
        for i in 0..m * n {
            let v1: T = rand::thread_rng().sample(&range);
            v.insert(i, v1);
        }
        Matrix { m: m, n: n, arr: v }
    }

    fn identity(m: usize, n: usize) -> Self {
        let one: T = T::one();

        let mut matrix: Matrix<T> = Matrix::new(m, n);

        let mut i = 0;
        for x in 0..m {
            matrix.arr[i + x * n] = one;
            i += 1;
        }

        matrix
    }

    fn invert(mut self) -> Option<Self> {
        if self.n != self.m {
            return None;
        }

        let mut _identity: Matrix<T> = Matrix::identity(self.m, self.n);
        println!("{:?}", _identity);
        println!("{:?}", self);

        // set ones diagonally else None

        // remove allother values and return Some

        println!("{:?}", _identity);
        println!("{:?}", self);
        Some(_identity)
    }
}

impl<T: Copy> Matrix<T> {
    fn transpose(&self) -> Self {
        let mut v: Vec<T> = vec![];
        for i in 0..self.n {
            for x in 0..self.m {
                v.push(self.arr[i + x * self.n]);
            }
        }
        Matrix {
            m: self.n,
            n: self.m,
            arr: v,
        }
    }

    fn get_row(&self, i: usize) -> Option<Vec<T>> {
        if i >= self.m {
            return None;
        }

        let start: usize = self.n * i;
        let v = self.arr[start..start + self.n].to_owned();

        Some(v)
    }

    fn get_row_mut(&mut self, i: usize) -> Option<&mut [T]> {
        if i >= self.m {
            return None;
        }

        let start: usize = self.n * i;
        let v = &mut self.arr[start..start + self.n];

        Some(v)
    }

    fn get_col(&self, i: usize) -> Option<Vec<T>> {
        if i >= self.n {
            return None;
        }

        let mut v = vec![];

        for x in 0..self.m {
            v.push(self.arr[i + x * self.n]);
        }

        Some(v)
    }
}

impl<T: Num + AddAssign + Copy> Add for Matrix<T> {
    type Output = Option<Self>;

    fn add(mut self, rhs: Self) -> Self::Output {
        if rhs.m != self.m || rhs.n != self.n {
            return None;
        }

        for i in 0..self.m * self.n {
            self.arr[i] += rhs.arr[i];
        }

        Some(self)
    }
}

impl<T: Num + SubAssign + Copy> Sub for Matrix<T> {
    type Output = Option<Self>;

    fn sub(mut self, rhs: Self) -> Self::Output {
        if rhs.m != self.m || rhs.n != self.n {
            return None;
        }

        for i in 0..self.m * self.n {
            self.arr[i] -= rhs.arr[i];
        }

        Some(self)
    }
}

impl<T: Num + AddAssign + Copy + Sum> Mul for Matrix<T> {
    type Output = Option<Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.n != rhs.m {
            return None;
        }

        let mut v: Vec<T> = vec![];

        for i in 0..self.m {
            for x in 0..rhs.n {
                let r = self.get_row(i).unwrap();
                let c = rhs.get_col(x).unwrap();

                let value: T = r
                    .iter()
                    .zip(c.iter())
                    .map(|(x1, y1)| x1.to_owned() * y1.to_owned())
                    .sum();

                v.push(value);
            }
        }

        Some(Matrix {
            m: self.m,
            n: rhs.n,
            arr: v,
        })
    }
}

impl<T: Copy + Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::from("");

        let first = self.arr[0];
        let second = self.arr[self.n - 1];
        let third = self.arr[self.n * (self.m - 1)];
        let fourth = self.arr[self.n * self.m - 1];

        s = s.add(&format!(
            "{}\t...\t{}\n...\t...\t...\n{}\t...\t{}",
            first, second, third, fourth,
        ));

        write!(f, "\nMatrix {} x {}\n{}\n", self.m, self.n, s)
    }
}

impl<T: Copy + Display> Debug for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::from("");

        for i in 0..self.m * self.n {
            if i % self.n == 0 {
                s = s.add("\n");
            }
            s = s.add(&format!("{}\t", self.arr[i]));
        }

        write!(f, "\nMatrix {} x {}{}\n", self.m, self.n, s)
    }
}

// impl Mul<Matrix<i32>> for i32 {
//     type Output = Matrix<i32>;
//     fn mul(self, mut rhs: Matrix<i32>) -> Matrix<i32> {
//         for val in rhs.arr.iter_mut() {
//             *val *= self;
//         }
//         rhs
//     }
// }

fn main() {
    let m: Matrix<i32> = Matrix::random(3, 3, -20, 20);
    println!("{:?}", m.invert());
    // let m2 = Matrix::random(2, 3, i32::default(), 5);
    // println!("{:?}", m2);
    // let m3 = m - m2;
    // println!("{:?}", m3);
}
