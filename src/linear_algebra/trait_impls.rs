use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Sub, SubAssign},
};

use super::Matrix;

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    /// (row, col)
    fn index(&self, index: (usize, usize)) -> &f64 {
        &self.arr[index.0 + self.c * index.1]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    /// (row, col)
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.arr[index.0 + self.c * index.1]
    }
}

impl AsRef<Matrix> for Matrix {
    fn as_ref(&self) -> &Matrix {
        self
    }
}

impl<T: AsRef<Matrix>> Add<T> for Matrix {
    type Output = Self;

    fn add(mut self, rhs: T) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<T: AsRef<Matrix>> AddAssign<T> for Matrix {
    fn add_assign(&mut self, rhs: T) {
        if rhs.as_ref().r != self.r || rhs.as_ref().c != self.c {
            panic!("Cannot add matrices of different shape");
        }

        for i in 0..self.r * self.c {
            self.arr[i] += rhs.as_ref().arr[i];
        }
    }
}

impl<T: AsRef<Matrix>> Sub<T> for Matrix {
    type Output = Self;

    fn sub(mut self, rhs: T) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl<T: AsRef<Matrix>> SubAssign<T> for Matrix {
    fn sub_assign(&mut self, rhs: T) {
        if rhs.as_ref().r != self.r || rhs.as_ref().c != self.c {
            panic!(
                "Cannot subtract matrices of different shape. shape1:{:?} - shape2:{:?}",
                rhs.as_ref().get_dims(),
                self.as_ref().get_dims()
            );
        }

        for i in 0..self.r * self.c {
            self.arr[i] -= rhs.as_ref().arr[i];
        }
    }
}

impl Mul<Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, mut rhs: Matrix) -> Self::Output {
        rhs.iter_mut().for_each(|e| *e *= self);
        rhs
    }
}

impl Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.iter_mut().for_each(|e| *e *= rhs);
        self
    }
}

impl Div<f64> for Matrix {
    type Output = Matrix;

    fn div(mut self, rhs: f64) -> Self::Output {
        self.iter_mut().for_each(|e| *e /= rhs);
        self
    }
}

impl<T: AsRef<Matrix>> Mul<T> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: T) -> Self::Output {
        let _rhs = rhs.as_ref();
        if self.c != _rhs.r {}

        let mut v: Vec<f64> = Vec::with_capacity(self.r * _rhs.c);
        unsafe { v.set_len(self.r * _rhs.c) }
        unsafe {
            cblas::dgemm(
                cblas::Layout::ColumnMajor,
                cblas::Transpose::None,
                cblas::Transpose::None,
                self.r as i32,
                _rhs.c as i32,
                self.c as i32,
                1.0,
                &self.arr,
                self.r as i32,
                &_rhs.arr,
                _rhs.r as i32,
                0.0,
                &mut v,
                self.r as i32,
            )
        }

        Matrix {
            r: self.r,
            c: rhs.as_ref().c,
            arr: v,
        }
    }
}

impl<T: AsRef<Matrix>> Mul<T> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: T) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::from("");

        let first = self.arr[0];
        let second = self.arr[self.c * (self.r - 1)];
        let third = self.arr[self.c - 1];
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

        let (mut ri, mut ci) = (0, 0);

        for i in 0..self.r * self.c {
            ci += 1;
            if i % self.c == 0 {
                ci = 1;
                ri += 1;
                s = s.add("\t\n\n");
            }
            let mod_i = match self[(ri - 1, ci - 1)] > 0.0 {
                true => String::from("+"),
                false => String::new(),
            };
            s = s.add(&format!("{}{:.2}\t", mod_i, self[(ri - 1, ci - 1)]));
        }

        write!(f, "\nMatrix {} x {}{}\n", self.r, self.c, s)
    }
}

#[cfg(test)]
mod tests {
    use crate::linear_algebra::Matrix;

    #[test]
    fn matrix_mult() {
        let a = Matrix::from_vec(4, 1, vec![1.0, 1.0, 1.0, 1.0]);
        let b = Matrix::from_vec(1, 4, vec![3.0, 1.0, 11.0, 0.0]);

        let c = Matrix::from_vec(
            4,
            4,
            vec![
                3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 11.0, 11.0, 11.0, 11.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );

        assert_eq!(c, (a * b), "Testing Matrix Mult 4,1x1,4");

        let a = Matrix::from_vec(3, 1, vec![1.0, 1.0, 1.0]);
        let b = Matrix::from_vec(1, 4, vec![3.0, 1.0, 11.0, 0.0]);

        let c = Matrix::from_vec(
            3,
            4,
            vec![
                3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 11.0, 11.0, 11.0, 0.0, 0.0, 0.0,
            ],
        );

        assert_eq!(c, (a * b), "Testing Matrix Mult 3,1x1,4");

        let a = Matrix::from_vec(3, 2, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let b = Matrix::from_vec(2, 2, vec![3.0, 1.0, 11.0, 0.0]);

        let c = Matrix::from_vec(3, 2, vec![4.0, 4.0, 4.0, 11.0, 11.0, 11.0]);

        assert_eq!(c, (a * b), "Testing Matrix Mult 3,2x2,2");

        let a = Matrix::from_vec(1, 4, vec![3.0, 1.0, 11.0, 0.0]);
        let b = Matrix::from_vec(4, 1, vec![1.0, 1.0, 1.0, 1.0]);

        let c = Matrix::from_vec(1, 1, vec![15.0]);

        assert_eq!(c, (a * b), "Testing Matrix Mult 1,4x4,1");
    }
}
