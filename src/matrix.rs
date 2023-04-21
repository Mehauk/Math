use num_traits::{Num, One};
use rand::{
    distributions::{uniform::SampleUniform, Uniform},
    Rng,
};
use std::{
    fmt::{Debug, Display},
    iter::Sum,
    ops::{Add, AddAssign, Index, Mul, Neg, Sub, SubAssign},
};

pub struct Matrix<T> {
    pub m: usize,
    pub n: usize,
    pub arr: Vec<T>,
}

impl<
        T: Num + Copy + Clone + SampleUniform + Default + One + Display + AddAssign + Neg<Output = T>,
    > Matrix<T>
{
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

    fn determinant(&self) -> T {
        if self.size() == 1 {
            return self[(0, 0)];
        }

        let mut multiplier = -T::one();

        let mut value = T::zero();

        if let Some(v) = self.get_row(0) {
            for i in 0..v.len() {
                multiplier = -multiplier;

                let mut arr: Vec<T> = vec![];

                for x in 0..self.n {
                    if x != i {
                        if let Some(c) = self.get_col(x) {
                            arr.append(&mut c[1..c.len()].to_vec());
                        }
                    }
                }

                let mat = Matrix {
                    m: self.m - 1,
                    n: self.n - 1,
                    arr: arr,
                };

                value += multiplier * (v[i]) * mat.determinant();
            }
        }

        value
    }

    pub fn invert(mut self) -> Option<Self> {
        if self.n != self.m {
            return None;
        }

        let mut _identity: Matrix<T> = Matrix::identity(self.m, self.n);
        println!("{:?}", _identity);
        println!("{:?}", self);

        // find determinant
        if self.determinant() == T::default() {
            return None;
        }

        // set ones diagonally
        for i in 0..self.n {
            if let Some(c) = self.get_col(i) {
                // find first non-zero
                let mut index = i;
                for x in i..self.m {
                    if c[x] != T::zero() {
                        index = x;
                        break;
                    }
                }

                // Swap rows
                if index != i {
                    println!("SADFASDFAS");
                    if let (Some(r1), Some(r2)) = (self.get_row(i), self.get_row(index)) {
                        let n = self.n;
                        if let Some(r1x) = self.get_row_mut(i) {
                            for v in 0..n {
                                r1x[v] = r1[v];
                            }
                        }
                        if let Some(r2x) = self.get_row_mut(index) {
                            for v in 0..n {
                                r2x[v] = r2[v];
                            }
                        }
                    };
                }
            }
        }

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

    fn size(&self) -> usize {
        self.m * self.n
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        &self.arr[index.1 + self.n * index.0]
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
