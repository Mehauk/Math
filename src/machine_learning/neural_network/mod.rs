use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::calculus::functions::Function;

use super::dataset::DataSet;

#[derive(Deserialize, Serialize, Debug)]
/// ### Parameters
/// - `_weights` : `Vec<DMatrix<f64>>`
/// - `hidden_layer` : `Vec<DVector<f64>>`
pub struct NueralNetwork {
    _weigths: Vec<DMatrix<f64>>,
    _biases: Vec<DVector<f64>>,
    _shape: Vec<usize>,
}

impl NueralNetwork {
    pub fn random(shape: Vec<usize>) -> NueralNetwork {
        let length = shape.len();
        let weigths: Vec<DMatrix<f64>> = shape[1..length]
            .iter()
            .zip(&shape[0..length - 1])
            .map(|(a, b)| random_matrix(*a, *b))
            .collect();
        let biases: Vec<DVector<f64>> =
            shape[1..length].iter().map(|a| random_vector(*a)).collect();

        NueralNetwork {
            _weigths: weigths,
            _biases: biases,
            _shape: shape,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> NueralNetwork {
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

        NueralNetwork {
            _weigths: weigths,
            _biases: biases,
            _shape: shape,
        }
    }

    fn step(&mut self, other: Self, learning_rate: f64) {
        for (a, b) in self._weigths.iter_mut().zip(other._weigths) {
            *a -= b * learning_rate;
        }

        for (a, b) in self._biases.iter_mut().zip(other._biases) {
            *a -= b * learning_rate;
        }
    }

    pub fn propagate(
        &self,
        input: &DVector<f64>,
        activation_function: fn(&mut f64),
    ) -> DVector<f64> {
        let mut propagating_nodes: &DVector<f64> = input;
        let mut x = DVector::zeros(0);

        for (weight_matrix, bias_vector) in self._weigths.iter().zip(&self._biases) {
            x = (weight_matrix * propagating_nodes + bias_vector).apply_into(activation_function);
            propagating_nodes = &x;
        }

        x
    }

    /// Returns a vector of intermediate node_vectors including the output vector and excluding the input vector
    fn propagate_returning_all_nodes_prior_to_activation(
        &self,
        input: &DVector<f64>,
        activation_function: fn(&mut f64),
    ) -> Vec<DVector<f64>> {
        // initialize resulting array;
        let mut nodes_array: Vec<DVector<f64>> = vec![];

        let mut propagating_nodes: DVector<f64> = input.clone();

        for (weight_matrix, bias_vector) in self._weigths.iter().zip(&self._biases) {
            nodes_array.push(weight_matrix * propagating_nodes + bias_vector);
            propagating_nodes = nodes_array
                .last()
                .unwrap()
                .clone()
                .apply_into(activation_function);
        }

        nodes_array
    }

    /// calculates the cost the nueral network; `C = (R - E)^2`
    /// - `C` cost Matrix
    /// - `R` resulting output Matrix of network
    /// - `E` expected output Matrix contructed from label
    pub fn _cost(result_matrix: &DVector<f64>, label: u8) -> DVector<f64> {
        let mut m = result_matrix.clone();
        m[(label as usize, 0)] -= 1.0;
        m.apply_into(|x| *x *= *x)
    }

    /// calculates the derivative of the cost; `C' = 2(R - E)
    /// - `C` cost Matrix
    /// - `R` resulting output Matrix of network
    /// - `E` expected output Matrix contructed from label
    pub fn cost_derivative(result_matrix: &DVector<f64>, label: u8) -> DVector<f64> {
        let mut m = result_matrix.clone();
        m[(label as usize, 0)] -= 1.0;
        m.scale(2.0)
    }

    // train using stochastic gradient descent
    pub fn train(
        &mut self,
        data_set: &DataSet,
        batch_size: usize,
        learning_rate: f64,
        activation_function: &Function,
    ) {
        println!("Training has commenced.");
        let data_set_length = data_set.training_data.len();
        let number_of_batches = data_set_length / batch_size;
        let remaining_data = data_set_length % batch_size;

        let mut loading_indicator: [char; 10] = ['_'; 10];
        print!(
            "\rTraining in progress: {} - {:0>3.2}% complete",
            loading_indicator.iter().collect::<String>(),
            0.0
        );
        for n in 0..number_of_batches {
            let fraction = n as f64 / number_of_batches as f64;
            loading_indicator[(fraction * 9.0) as usize] = '█';
            self.step(
                self.calculate_batch_step(
                    data_set,
                    n * batch_size,
                    (n + 1) * batch_size,
                    activation_function,
                )
                .unwrap(),
                learning_rate,
            );

            print!(
                "\rTraining in progress: {} - {:0>3.2}% complete",
                loading_indicator.iter().collect::<String>(),
                fraction * 100.0,
            );
        }
        self.step(
            self.calculate_batch_step(
                data_set,
                number_of_batches * batch_size,
                number_of_batches * batch_size + remaining_data,
                activation_function,
            )
            .unwrap(),
            learning_rate,
        );

        loading_indicator[9] = '█';
        print!(
            "\rTraining in progress: {} - {:0>3.2}% complete\n",
            loading_indicator.iter().collect::<String>(),
            100.0,
        );
    }

    fn calculate_batch_step(
        &self,
        data_set: &DataSet,
        batch_start: usize,
        batch_end: usize,
        activation_function: &Function,
    ) -> Option<Self> {
        let mut delta_network = NueralNetwork::zeros(self._shape.clone());
        for i in batch_start..batch_end {
            let training_data = &data_set.training_data[i];
            let mut nodes = self.propagate_returning_all_nodes_prior_to_activation(
                &training_data.data,
                activation_function.activate,
            );

            let nodes_cur = nodes.pop()?;
            let mut index = nodes.len();

            let mut delta_cost_by_delta_nodes = NueralNetwork::cost_derivative(
                &nodes_cur.clone().apply_into(activation_function.activate),
                training_data.label,
            );

            let mut delta_nodes_by_delta_activation =
                nodes_cur.apply_into(activation_function.derive);

            loop {
                let nodes_cur = nodes.pop().unwrap_or(training_data.data.clone());
                let weights_cur = &self._weigths[index];

                // calculate and store bias delta for each layer
                let delta_cost_by_delta_biases =
                    delta_cost_by_delta_nodes.component_mul(&delta_nodes_by_delta_activation);
                delta_network._biases[index] = delta_cost_by_delta_biases;

                // calculate and store weight delta for each layer
                let delta_cost_by_delta_weights = &delta_network._biases[index]
                    * nodes_cur
                        .clone()
                        .apply_into(activation_function.activate)
                        .transpose();

                if index == 0 {
                    delta_network._weigths[index] =
                        &delta_network._biases[index] * nodes_cur.transpose();
                    break;
                }

                delta_network._weigths[index] = delta_cost_by_delta_weights;

                // calculate new delta for nodes
                delta_cost_by_delta_nodes = weights_cur.transpose() * &delta_network._biases[index];

                // calculate new activation function delta
                delta_nodes_by_delta_activation = nodes_cur.apply_into(activation_function.derive);

                index -= 1;
            }
        }

        Some(delta_network)
    }

    pub fn test(&self, data_set: &DataSet, activation_function: &Function) -> f64 {
        let data_set_length = data_set.testing_data.len() as f64;
        let mut total_correct = 0.0;

        for image in data_set.testing_data.iter() {
            let res = self.propagate(&image.data, activation_function.activate);
            if res.column(0).argmax().0 == (image.label as usize) {
                total_correct += 1.0;
            }
        }

        total_correct / data_set_length
    }
}

fn random_matrix(r: usize, c: usize) -> DMatrix<f64> {
    DMatrix::<f64>::from_distribution(
        r,
        c,
        &rand::distributions::Standard,
        &mut rand::thread_rng(),
    )
}

fn random_vector(r: usize) -> DVector<f64> {
    DVector::<f64>::from_distribution(r, &rand::distributions::Standard, &mut rand::thread_rng())
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::{
        calculus::functions::Function,
        machine_learning::dataset::{DataSet, DataVector},
    };

    use super::NueralNetwork;

    fn init_network(v: Vec<usize>) -> (NueralNetwork, DataSet) {
        let mut nn = NueralNetwork::zeros(v);
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
    fn test_both_propagation_methods_are_equivalent() {
        let (nn, ds) = init_network(vec![1, 2, 2]);

        let f = Function::sigmoid();

        let x = nn
            .propagate_returning_all_nodes_prior_to_activation(
                &ds.training_data.first().unwrap().data,
                f.activate,
            )
            .last()
            .unwrap()
            .clone()
            .apply_into(f.activate);
        let y = nn.propagate(&ds.training_data.first().unwrap().data, f.activate);

        assert_eq!(x, y);
    }

    #[test]
    fn test_multiple_propagation_calls() {
        let (nn, ds) = init_network(vec![1, 2, 2]);
        let f = Function::sigmoid();

        let y = nn.propagate(&ds.training_data.first().unwrap().data, f.activate);
        let x = nn.propagate(&ds.training_data.first().unwrap().data, f.activate);

        assert!(x.len() == 2);
        assert!(y.len() == 2);
        assert_eq!(x, y);
    }

    #[test]
    fn test_cost() {
        let (nn, ds) = init_network(vec![1, 2, 2]);
        let f = Function::sigmoid();

        let output = nn.propagate(&ds.testing_data.first().unwrap().data, f.activate);
        let cost = NueralNetwork::_cost(&output, 1);

        assert!(cost[0] - 0.3 < 0.01);
        assert!(cost[1] - 0.2 < 0.01);
    }

    #[test]
    fn test_cost_derivative() {
        let (nn, ds) = init_network(vec![1, 2, 2]);
        let f = Function::sigmoid();

        let output = nn.propagate(&ds.testing_data.first().unwrap().data, f.activate);
        let derivative = NueralNetwork::cost_derivative(&output, 1);

        assert!(derivative[0] - 1.1 < 0.01);
        assert!(derivative[1] + 0.9 < 0.01);
    }

    #[test]
    fn test_batch_step() {
        let (nn, ds) = init_network(vec![1, 2, 2]);
        let batch_step = nn.calculate_batch_step(&ds, 0, 1, &Function::sigmoid());

        match batch_step {
            Some(step) => {
                assert!(step._biases[0][0] - 0.0012 < 0.0001);
                assert!(step._biases[0][1] - 0.0012 < 0.0001);

                assert!(step._weigths[0][0] == 0.0);
                assert!(step._weigths[0][1] == 0.0);

                assert!(step._biases[1][0] - 0.27 < 0.01);
                assert!(step._biases[1][1] - 0.27 < 0.01);

                assert!(step._weigths[1][0] - 0.14 < 0.01);
                assert!(step._weigths[1][1] + 0.11 < 0.01);
                assert!(step._weigths[1][2] - 0.14 < 0.01);
                assert!(step._weigths[1][3] + 0.11 < 0.01);
            }
            None => panic!("No batch step calculated."),
        }
    }

    #[test]
    fn test_training_sigmoid() {
        let (mut nn, ds) = init_network(vec![1, 3, 2]);

        let f = Function::sigmoid();

        nn.train(&ds, 1, 1.0, &f);

        assert!(
            0 == nn
                .propagate(&DVector::from_vec(vec![0.7]), f.activate)
                .argmax()
                .0
        );
        assert!(
            1 == nn
                .propagate(&DVector::from_vec(vec![0.4]), f.activate)
                .argmax()
                .0
        );
    }

    #[test]
    fn test_training_normal_arctan() {
        let (mut nn, ds) = init_network(vec![1, 3, 2]);

        let f = Function::normal_arctan();

        nn.train(&ds, 1, 1.0, &f);

        assert!(
            0 == nn
                .propagate(&DVector::from_vec(vec![0.7]), f.activate)
                .argmax()
                .0
        );
        assert!(
            1 == nn
                .propagate(&DVector::from_vec(vec![0.4]), f.activate)
                .argmax()
                .0
        );
    }
}
