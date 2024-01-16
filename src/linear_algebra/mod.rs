use rand::{distributions::Distribution, Rng};

pub mod methods;
pub mod trait_impls;

#[derive(PartialEq, Clone)]
pub struct Matrix {
    r: usize,
    c: usize,
    arr: Vec<f64>,
}

impl Matrix {
    pub fn from_value(r: usize, c: usize, val: f64) -> Self {
        let v = vec![val; c * r];
        Matrix { r, c, arr: v }
    }

    pub fn zeros(r: usize, c: usize) -> Self {
        Self::from_value(r, c, 0.0)
    }

    pub fn from_vec(r: usize, c: usize, mut arr: Vec<f64>) -> Self {
        while arr.len() < r * c {
            arr.push(0.0);
        }
        Matrix { r, c, arr }
    }

    pub fn from_iterator(r: usize, c: usize, iter: &mut dyn Iterator<Item = f64>) -> Self {
        let mut arr: Vec<f64> = iter.collect();
        while arr.len() < r * c {
            arr.push(0.0);
        }
        Matrix { r, c, arr }
    }

    pub fn from_distribution(r: usize, c: usize, distribution: &impl Distribution<f64>) -> Self {
        let mut v: Vec<f64> = (0..r * c)
            .map(|_| rand::thread_rng().sample(&distribution))
            .collect();
        Matrix { r, c, arr: v }
    }

    pub fn identity(r: usize, c: usize) -> Self {
        let mut matrix: Matrix = Matrix::zeros(r, c);

        let mut i = 0;
        for x in 0..r {
            matrix.arr[i + x * c] = 1.0;
            i += 1;
        }

        matrix
    }
}
