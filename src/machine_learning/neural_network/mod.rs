use std::{error::Error, io};

use nalgebra::{DMatrix, DVector};

pub mod methods;

#[derive(Debug, PartialEq)]
/// ### Parameters
/// - `_weights` : `Vec<DMatrix<f64>>`
/// - `hidden_layer` : `Vec<DVector<f64>>`
pub struct NeuralNetwork {
    _weigths: Vec<DMatrix<f64>>,
    _biases: Vec<DVector<f64>>,
    _shape: Vec<usize>,
}

// NN contructors / destructors
impl NeuralNetwork {
    pub fn random(shape: Vec<usize>) -> NeuralNetwork {
        let length = shape.len();
        let weigths: Vec<DMatrix<f64>> = shape[1..length]
            .iter()
            .zip(&shape[0..length - 1])
            .map(|(a, b)| DMatrix::new_random(*a, *b))
            .collect();
        let biases: Vec<DVector<f64>> = shape[1..length]
            .iter()
            .map(|a| DVector::new_random(*a))
            .collect();

        NeuralNetwork {
            _weigths: weigths,
            _biases: biases,
            _shape: shape,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> NeuralNetwork {
        let length = shape.len();
        let weigths: Vec<DMatrix<f64>> = shape[1..length]
            .iter()
            .zip(&shape[0..length - 1])
            .map(|(a, b)| DMatrix::<f64>::zeros(*a, *b))
            .collect();
        let biases: Vec<DVector<f64>> = shape[1..length]
            .iter()
            .map(|a| DVector::<f64>::zeros(*a))
            .collect();

        NeuralNetwork {
            _weigths: weigths,
            _biases: biases,
            _shape: shape,
        }
    }
}

// NN save / load
impl NeuralNetwork {
    pub fn save(&self, file_path: &str) -> Result<(), io::Error> {
        let mut contents = String::new();

        contents += &self
            ._shape
            .iter()
            .map(|x| x.to_string())
            .collect::<String>();
        contents += "\n\n";

        for w in self._weigths.iter() {
            contents += &format!(
                "{},{} - {}\n",
                w.shape().0,
                w.shape().1,
                w.data
                    .as_vec()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(","),
            )
            .to_string();
        }

        contents += "\n";

        for b in self._biases.iter() {
            contents += &format!(
                "{} - {}\n",
                b.shape().0,
                b.data
                    .as_vec()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(","),
            )
            .to_string();
        }

        std::fs::write(file_path, contents)?;
        Ok(())
    }

    pub fn load(file_path: &str) -> Result<NeuralNetwork, Box<dyn Error>> {
        let contents = std::str::from_utf8(&std::fs::read(file_path)?)?;

        let c_split = contents.split("\n\n");
        let shape = c_split
            .next()
            .ok_or("File was not properly formatted, or was empty")?;
        let w = c_split
            .next()
            .ok_or("File was not properly formatted, or was empty")?;
        let b = c_split
            .next()
            .ok_or("File was not properly formatted, or was empty")?;

        let mut weights: Vec<DMatrix<f64>> = Vec::new();
        for w_line in w.split("\n") {
            let iter = w_line.split(" - ");
            let shape = iter.next().ok_or("Incorrect Format")?.split(",");
            let vals = iter
                .next()
                .ok_or("Incorrect Format")?
                .split(",")
                .map(|x| x.parse().unwrap_or_default());
            weights.push(DMatrix::<f64>::from_iterator(
                shape.next().ok_or("Incorrect Format")?.parse()?,
                shape.next().ok_or("Incorrect Format")?.parse()?,
                vals,
            ))
        }

        let mut weights: Vec<DVector<f64>> = Vec::new();
        for b_line in b.split("\n") {}

        Ok(NeuralNetwork {
            _weigths: weights,
            _biases: biases,
            _shape: shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::{
        calculus::functions::Function,
        machine_learning::dataset::{DataSet, DataVector},
    };

    use super::NeuralNetwork;

    pub fn init_network(v: Vec<usize>) -> (NeuralNetwork, DataSet) {
        let mut nn = NeuralNetwork::zeros(v);
        for bv in nn._biases.iter_mut() {
            bv.add_scalar_mut(0.1);
        }

        for wv in nn._weigths.iter_mut() {
            wv.add_scalar_mut(0.1);
        }

        let ds = DataSet {
            training_data: (0..10000)
                .map(|x| {
                    let y: f64 = rand::random::<f64>() * x as f64 / (x + 1) as f64;
                    DataVector::_new(DVector::from_vec(vec![y]), if y < 0.5 { 1 } else { 0 })
                })
                .collect(),
            testing_data: (0..100)
                .map(|x: usize| {
                    let y: f64 = x as f64 / 100.0;
                    DataVector::_new(DVector::from_vec(vec![y]), if y < 0.5 { 1 } else { 0 })
                })
                .collect(),
        };

        return (nn, ds);
    }

    #[test]
    fn test_saving_and_loading() {
        let (nn, _) = init_network(vec![1, 3, 2]);
        let file = "output/before.nn";

        nn.save(file).unwrap();
        let snn = NeuralNetwork::load(file).unwrap();

        assert!(snn == nn);
    }
}
