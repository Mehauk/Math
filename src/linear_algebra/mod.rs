use rand::{distributions::Distribution, Rng};

pub mod methods;
pub mod trait_impls;

#[derive(PartialEq)]
pub struct Matrix {
    m: usize,
    n: usize,
    arr: Vec<f64>,
}

impl Matrix {
    pub fn zeros(m: usize, n: usize) -> Self {
        let v = vec![0.0; n * m];
        Matrix { m, n, arr: v }
    }

    pub fn from_vec(m: usize, n: usize, mut arr: Vec<f64>) -> Self {
        while arr.len() < m * n {
            arr.push(0.0);
        }
        Matrix { m, n, arr }
    }

    pub fn from_distribution(m: usize, n: usize, distribution: &impl Distribution<f64>) -> Self {
        let mut v: Vec<f64> = (0..m * n)
            .map(|_| rand::thread_rng().sample(&distribution))
            .collect();
        Matrix { m, n, arr: v }
    }

    pub fn identity(m: usize, n: usize) -> Self {
        let mut matrix: Matrix = Matrix::zeros(m, n);

        let mut i = 0;
        for x in 0..m {
            matrix.arr[i + x * n] = 1.0;
            i += 1;
        }

        matrix
    }
}
