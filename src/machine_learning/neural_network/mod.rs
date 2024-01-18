use std::{error::Error, io};

use rand::distributions::Uniform;

use crate::linear_algebra::Matrix;

pub mod methods;

#[derive(Debug, PartialEq)]
/// ### Parameters
/// - `_weights` : `Vec<Matrix>`
/// - `hidden_layer` : `Vec<Matrix>`
pub struct NeuralNetwork {
    _weigths: Vec<Matrix>,
    _biases: Vec<Matrix>,
    _shape: Vec<usize>,
}

// NN contructors / destructors
impl NeuralNetwork {
    pub fn random(shape: Vec<usize>) -> NeuralNetwork {
        let length = shape.len();
        let weigths: Vec<Matrix> = shape[1..length]
            .iter()
            .zip(&shape[0..length - 1])
            .map(|(a, b)| Matrix::from_distribution(*a, *b, &Uniform::new(-1.0, 1.0)))
            .collect();
        let biases: Vec<Matrix> = shape[1..length]
            .iter()
            .map(|a| Matrix::from_distribution(*a, 1, &Uniform::new(-1.0, 1.0)))
            .collect();

        NeuralNetwork {
            _weigths: weigths,
            _biases: biases,
            _shape: shape,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> NeuralNetwork {
        NeuralNetwork::scalar(shape, 0.0)
    }

    pub fn scalar(shape: Vec<usize>, scale: f64) -> NeuralNetwork {
        let length = shape.len();
        let weigths: Vec<Matrix> = shape[1..length]
            .iter()
            .zip(&shape[0..length - 1])
            .map(|(a, b)| Matrix::from_value(*a, *b, scale))
            .collect();
        let biases: Vec<Matrix> = shape[1..length]
            .iter()
            .map(|a| Matrix::from_value(*a, 1, scale))
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
            .collect::<Vec<String>>()
            .join(",");
        contents += "\n\n";

        for w in self._weigths.iter() {
            contents += &w.to_str();
        }

        contents += "\n";

        for b in self._biases.iter() {
            contents += &b.to_str();
        }

        contents += "\n";

        std::fs::write(file_path, contents)?;
        Ok(())
    }

    pub fn load(file_path: &str) -> Result<NeuralNetwork, Box<dyn Error>> {
        let binding = std::fs::read(file_path)?;
        let contents = std::str::from_utf8(&binding)?;

        let mut c_split = contents.split("\n\n");
        let shape = c_split
            .next()
            .ok_or("File was not properly formatted, or was empty")?
            .split(",")
            .map(|x| x.parse().unwrap_or(1))
            .collect::<Vec<usize>>();
        let w = c_split
            .next()
            .ok_or("File was not properly formatted, or was empty")?;
        let b = c_split
            .next()
            .ok_or("File was not properly formatted, or was empty")?;

        let mut weights: Vec<Matrix> = Vec::new();
        for w_line in w.split("\n") {
            let mut iter = w_line.split(" - ");
            let mut shape = iter.next().ok_or("Incorrect Format weights")?.split(",");
            let mut vals = iter
                .next()
                .ok_or("Incorrect Format for wieght values")?
                .split(",")
                .map(|x| x.parse().unwrap_or_default());
            weights.push(Matrix::from_iterator(
                shape
                    .next()
                    .ok_or("Incorrect Format for weights rows")?
                    .parse()?,
                shape
                    .next()
                    .ok_or("Incorrect Format for weights cols")?
                    .parse()?,
                &mut vals,
            ))
        }

        let mut biases: Vec<Matrix> = Vec::new();
        for b_line in b.split("\n") {
            let mut iter = b_line.split(" - ");
            let mut shape = iter.next().ok_or("Incorrect Format biases")?.split(",");
            let mut vals = iter
                .next()
                .ok_or("Incorrect Format for bias values")?
                .split(",")
                .map(|x| x.parse().unwrap_or_default());
            biases.push(Matrix::from_iterator(
                shape
                    .next()
                    .ok_or("Incorrect Format for weights rows")?
                    .parse()?,
                shape
                    .next()
                    .ok_or("Incorrect Format for weights cols")?
                    .parse()?,
                &mut vals,
            ))
        }

        Ok(NeuralNetwork {
            _weigths: weights,
            _biases: biases,
            _shape: shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::{
        linear_algebra::Matrix,
        machine_learning::dataset::{DataSet, DataVector},
    };

    use super::NeuralNetwork;

    pub fn init_network(v: Vec<usize>) -> (NeuralNetwork, DataSet) {
        let nn = NeuralNetwork::scalar(v, 0.1);

        let ds = DataSet {
            training_data: (0..10000)
                .map(|x| {
                    let y: f64 = rand::random::<f64>() * x as f64 / (x + 1) as f64;
                    DataVector::_new(Matrix::from_vec(1, 1, vec![y]), if y < 0.5 { 1 } else { 0 })
                })
                .collect(),
            testing_data: (0..100)
                .map(|x: usize| {
                    let y: f64 = x as f64 / 100.0;
                    DataVector::_new(Matrix::from_vec(1, 1, vec![y]), if y < 0.5 { 1 } else { 0 })
                })
                .collect(),
        };

        return (nn, ds);
    }

    #[test]
    fn test_saving_and_loading() {
        let (nn, _) = init_network(vec![12, 222, 21]);
        let file = "output/before.nn";

        nn.save(file).unwrap();
        let snn = NeuralNetwork::load(file).unwrap();

        assert!(snn == nn);
        fs::remove_file(file).unwrap();
    }
}
