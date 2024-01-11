use std::{
    fmt::{Debug, Display},
    ops::{Add, Div, Index, Mul, Sub},
};

use super::Matrix;

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &f64 {
        &self.arr[index.1 + self.c * index.0]
    }
}

impl Add for Matrix {
    type Output = Option<Self>;

    fn add(mut self, rhs: Self) -> Self::Output {
        if rhs.r != self.r || rhs.c != self.c {
            return None;
        }

        for i in 0..self.r * self.c {
            self.arr[i] += rhs.arr[i];
        }

        Some(self)
    }
}

impl Sub for Matrix {
    type Output = Option<Self>;

    fn sub(mut self, rhs: Self) -> Self::Output {
        if rhs.r != self.r || rhs.c != self.c {
            return None;
        }

        for i in 0..self.r * self.c {
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
        if self.c != rhs.r {
            return None;
        }

        let mut v: Vec<f64> = vec![];

        for i in 0..self.r {
            for x in 0..rhs.c {
                let r = self.get_row(i).unwrap();
                let c = rhs.get_col(x).unwrap();

                let value: f64 = r.iter().zip(c.iter()).map(|(x1, y1)| x1 * y1).sum();

                v.push(value);
            }
        }

        Some(Matrix {
            r: self.r,
            c: rhs.c,
            arr: v,
        })
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::from("");

        let first = self.arr[0];
        let second = self.arr[self.c - 1];
        let third = self.arr[self.c * (self.r - 1)];
        let fourth = self.arr[self.c * self.r - 1];

        s = s.add(&format!(
            "{:.2}\t...\t{:.2}\n...\t...\t...\n{:.2}\t...\t{:.2}",
            first, second, third, fourth,
        ));

        write!(f, "\nMatrix {} x {}\n{}\n", self.r, self.c, s)
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::from("");

        for i in 0..self.r * self.c {
            if i % self.c == 0 {
                s = s.add("\t\n\n");
            }
            let mod_i = match self.arr[i] > 0.0 {
                true => String::from("+"),
                false => String::new(),
            };
            s = s.add(&format!("{}{:.2}\t", mod_i, self.arr[i]));
        }

        write!(f, "\nMatrix {} x {}{}\n", self.r, self.c, s)
    }
}
