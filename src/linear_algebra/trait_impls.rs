use std::{
    fmt::{Debug, Display},
    ops::{Add, Div, Index, Mul, Sub},
};

use super::Matrix;

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &f64 {
        &self.arr[index.1 + self.n * index.0]
    }
}

impl Add for Matrix {
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

impl Sub for Matrix {
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

impl Mul<Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, mut rhs: Matrix) -> Self::Output {
        rhs.arr.iter_mut().for_each(|e| *e *= self);
        rhs
    }
}

impl Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.arr.iter_mut().for_each(|e| *e *= rhs);
        self
    }
}

impl Div<f64> for Matrix {
    type Output = Matrix;

    fn div(mut self, rhs: f64) -> Self::Output {
        self.arr.iter_mut().for_each(|e| *e /= rhs);
        self
    }
}

impl Mul for Matrix {
    type Output = Option<Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.n != rhs.m {
            return None;
        }

        let mut v: Vec<f64> = vec![];

        for i in 0..self.m {
            for x in 0..rhs.n {
                let r = self.get_row(i).unwrap();
                let c = rhs.get_col(x).unwrap();

                let value: f64 = r.iter().zip(c.iter()).map(|(x1, y1)| x1 * y1).sum();

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

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::from("");

        let first = self.arr[0];
        let second = self.arr[self.n - 1];
        let third = self.arr[self.n * (self.m - 1)];
        let fourth = self.arr[self.n * self.m - 1];

        s = s.add(&format!(
            "{:.2}\t...\t{:.2}\n...\t...\t...\n{:.2}\t...\t{:.2}",
            first, second, third, fourth,
        ));

        write!(f, "\nMatrix {} x {}\n{}\n", self.m, self.n, s)
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::from("");

        for i in 0..self.m * self.n {
            if i % self.n == 0 {
                s = s.add("\t\n\n");
            }
            let mod_i = match self.arr[i] > 0.0 {
                true => String::from("+"),
                false => String::new(),
            };
            s = s.add(&format!("{}{:.2}\t", mod_i, self.arr[i]));
        }

        write!(f, "\nMatrix {} x {}{}\n", self.m, self.n, s)
    }
}
