use std::ptr::swap_nonoverlapping;

use super::Matrix;

// Commonly used methods
impl Matrix {
    pub fn transpose(&self) -> Self {
        let mut v: Vec<f64> = vec![];
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

    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.arr.iter()
    }
}

// Extra methods
impl Matrix {
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

                let mat = Matrix {
                    m: self.m - 1,
                    n: self.n - 1,
                    arr,
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

    pub fn invert(mut self) -> Option<Matrix> {
        if self.n != self.m {
            return None;
        }

        let mut identity: Matrix = Matrix::identity(self.m, self.n);

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

    pub fn get_row(&self, i: usize) -> Option<Vec<f64>> {
        if i >= self.m {
            return None;
        }

        let start: usize = self.n * i;
        let v = self.arr[start..start + self.n].to_owned();

        Some(v)
    }

    pub fn get_col(&self, i: usize) -> Option<Vec<f64>> {
        if i >= self.n {
            return None;
        }

        let mut v = vec![];

        for x in 0..self.m {
            v.push(self.arr[i + x * self.n]);
        }

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

    fn size(&self) -> usize {
        self.m * self.n
    }
}
