use rand::{distributions::Uniform, Rng};

pub mod methods;
pub mod trait_impls;

#[derive(PartialEq)]
pub struct Matrix {
    m: usize,
    n: usize,
    arr: Vec<f64>,
}

impl Matrix {
    fn new(m: usize, n: usize) -> Self {
        let v = vec![0.0; n * m];
        Matrix { m, n, arr: v }
    }

    pub fn from(m: usize, n: usize, mut arr: Vec<f64>) -> Self {
        while arr.len() < m * n {
            arr.push(0.0);
        }
        Matrix { m, n, arr }
    }

    pub fn random(m: usize, n: usize, min: f64, max: f64) -> Self {
        let mut v: Vec<f64> = vec![];
        let range = Uniform::new(min, max);
        for i in 0..m * n {
            let v1: f64 = rand::thread_rng().sample(&range);
            v.insert(i, v1);
        }
        Matrix { m, n, arr: v }
    }

    pub fn identity(m: usize, n: usize) -> Self {
        let mut matrix: Matrix = Matrix::new(m, n);

        let mut i = 0;
        for x in 0..m {
            matrix.arr[i + x * n] = 1.0;
            i += 1;
        }

        matrix
    }
}
