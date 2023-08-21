use rand::{distributions::Uniform, Rng};
use std::{
    fmt::{Debug, Display},
    ops::{Add, Div, Index, Mul, Sub},
    ptr::swap_nonoverlapping,
};

pub struct CustomMatrix {
    m: usize,
    n: usize,
    arr: Vec<f64>,
}

impl CustomMatrix {
    fn new(m: usize, n: usize) -> Self {
        let v = vec![0.0; n * m];
        CustomMatrix { m: m, n: n, arr: v }
    }

    pub fn from(m: usize, n: usize, mut arr: Vec<f64>) -> Self {
        while arr.len() < m * n {
            arr.push(0.0);
        }
        CustomMatrix { m, n, arr }
    }

    pub fn random(m: usize, n: usize, min: f64, max: f64) -> Self {
        let mut v: Vec<f64> = vec![];
        let range = Uniform::new(min, max);
        for i in 0..m * n {
            let v1: f64 = rand::thread_rng().sample(&range);
            v.insert(i, v1);
        }
        CustomMatrix { m: m, n: n, arr: v }
    }

    pub fn identity(m: usize, n: usize) -> Self {
        let mut matrix: CustomMatrix = CustomMatrix::new(m, n);

        let mut i = 0;
        for x in 0..m {
            matrix.arr[i + x * n] = 1.0;
            i += 1;
        }

        matrix
    }

    pub fn copy(&self) -> Self {
        CustomMatrix {
            m: self.m,
            n: self.n,
            arr: self.arr.to_owned(),
        }
    }

    fn determinant(&self) -> f64 {
        if self.size() == 1 {
            return self[(0, 0)];
        }

        let mut multiplier = -1.0;

        let mut value = 0.0;

        if let Some(v) = self.get_row(0) {
            for (i, v) in v.iter().enumerate() {
                multiplier = -multiplier;

                let mut arr: Vec<f64> = vec![];

                for x in 0..self.n {
                    if x != i {
                        if let Some(c) = self.get_col(x) {
                            arr.append(&mut c[1..c.len()].to_vec());
                        }
                    }
                }

                let mat = CustomMatrix {
                    m: self.m - 1,
                    n: self.n - 1,
                    arr: arr,
                };

                value += multiplier * (v) * mat.determinant();
            }
        }
        value
    }

    fn swap_rows(&mut self, i0: usize, i1: usize) {
        unsafe {
            swap_nonoverlapping(
                self.get_row_mut(i0).unwrap().as_mut_ptr(),
                self.get_row_mut(i1).unwrap().as_mut_ptr(),
                self.n,
            )
        }
    }

    pub fn invert(mut self) -> Option<CustomMatrix> {
        if self.n != self.m {
            return None;
        }

        let mut identity: CustomMatrix = CustomMatrix::identity(self.m, self.n);

        // find determinant
        if self.determinant() == f64::default() {
            return None;
        }

        // swap rows
        for i in 0..self.n {
            if let Some(c) = self.get_col(i) {
                let mut i2 = i;
                for x in i..self.m {
                    if c[x] != 0.0 {
                        i2 = x;
                        break;
                    }
                }

                if i2 != i {
                    self.swap_rows(i, i2);
                    identity.swap_rows(i, i2);
                }
            }

            // normalize
            if let (Some(r), Some(ri)) = (self.get_row_mut(i), identity.get_row_mut(i)) {
                let d = r[i];
                if d != 0.0 {
                    r.iter_mut().for_each(|e| *e = *e / d);
                    ri.iter_mut().for_each(|e| *e = *e / d);
                }
            };

            if let (Some(r), Some(ri)) = (self.get_row(i), identity.get_row(i)) {
                // reduce all other values in column to zero
                for x in 0..self.m {
                    if i != x {
                        if let (Some(rc), Some(rci)) =
                            (self.get_row_mut(x), identity.get_row_mut(x))
                        {
                            let v = rc[i];
                            rc.iter_mut()
                                .zip(r.iter())
                                .for_each(|(x1, y1)| *x1 -= y1 * v);
                            rci.iter_mut()
                                .zip(ri.iter())
                                .for_each(|(x1, y1)| *x1 -= y1 * v);
                        }
                    }
                }
            }
        }

        // return the modified indentity matrix
        Some(identity)
    }

    pub fn transpose(&self) -> Self {
        let mut v: Vec<f64> = vec![];
        for i in 0..self.n {
            for x in 0..self.m {
                v.push(self.arr[i + x * self.n]);
            }
        }
        CustomMatrix {
            m: self.n,
            n: self.m,
            arr: v,
        }
    }

    fn get_row(&self, i: usize) -> Option<Vec<f64>> {
        if i >= self.m {
            return None;
        }

        let start: usize = self.n * i;
        let v = self.arr[start..start + self.n].to_owned();

        Some(v)
    }

    fn get_row_mut(&mut self, i: usize) -> Option<&mut [f64]> {
        if i >= self.m {
            return None;
        }

        let start: usize = self.n * i;
        let v = &mut self.arr[start..start + self.n];

        Some(v)
    }

    fn get_col(&self, i: usize) -> Option<Vec<f64>> {
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

impl Index<(usize, usize)> for CustomMatrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &f64 {
        &self.arr[index.1 + self.n * index.0]
    }
}

impl Add for CustomMatrix {
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

impl Sub for CustomMatrix {
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

impl Mul<CustomMatrix> for f64 {
    type Output = CustomMatrix;

    fn mul(self, mut rhs: CustomMatrix) -> Self::Output {
        rhs.arr.iter_mut().for_each(|e| *e *= self);
        rhs
    }
}

impl Mul<f64> for CustomMatrix {
    type Output = CustomMatrix;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.arr.iter_mut().for_each(|e| *e *= rhs);
        self
    }
}

impl Div<f64> for CustomMatrix {
    type Output = CustomMatrix;

    fn div(mut self, rhs: f64) -> Self::Output {
        self.arr.iter_mut().for_each(|e| *e /= rhs);
        self
    }
}

impl Mul for CustomMatrix {
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

        Some(CustomMatrix {
            m: self.m,
            n: rhs.n,
            arr: v,
        })
    }
}

impl Display for CustomMatrix {
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

impl Debug for CustomMatrix {
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
