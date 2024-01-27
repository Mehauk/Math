use std::{iter::Map, slice::Chunks};

use crate::{
    calculus::functions::Function,
    linear_algebra::Matrix,
    machine_learning::dataset::{DataSet, DataVector},
};

use rayon::prelude::*;

use super::NeuralNetwork;

// NN Methods
impl NeuralNetwork {
    fn step(&mut self, other: Self, learning_rate: f64) {
        for (a, b) in self._weigths.iter_mut().zip(other._weigths) {
            *a -= b * learning_rate;
        }

        for (a, b) in self._biases.iter_mut().zip(other._biases) {
            *a -= b * learning_rate;
        }
    }

    pub fn propagate(&self, input: &Matrix, activation_function: fn(&mut f64)) -> Matrix {
        let mut propagating_nodes: &Matrix = input;
        let mut x = Matrix::zeros(0, 0);

        for (weight_matrix, bias_vector) in self._weigths.iter().zip(&self._biases) {
            x = (weight_matrix * propagating_nodes + bias_vector).apply_into(activation_function);
            propagating_nodes = &x;
        }

        x
    }

    /// Returns a vector of intermediate node_vectors including the output vector and excluding the input vector
    fn propagate_returning_all_nodes_prior_to_activation(
        &self,
        input: &Matrix,
        activation_function: fn(&mut f64),
    ) -> Vec<Matrix> {
        // initialize resulting array;
        let mut nodes_array: Vec<Matrix> = vec![];

        let mut propagating_nodes: Matrix = input.clone();

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
    pub fn _cost(result_matrix: &Matrix, label: u8) -> Matrix {
        let mut m = result_matrix.clone();
        m[(label as usize, 0)] -= 1.0;
        m.apply_into(|x| *x *= *x)
    }

    /// calculates the derivative of the cost; `C' = 2(R - E)
    /// - `C` cost Matrix
    /// - `R` resulting output Matrix of network
    /// - `E` expected output Matrix contructed from label
    pub fn cost_derivative(result_matrix: &Matrix, label: u8) -> Matrix {
        let mut m = result_matrix.clone();
        m[(label as usize, 0)] -= 1.0;
        m = m * 2.0;
        m
    }

    // train using stochastic gradient descent
    pub fn train<'a>(
        &'a mut self,
        data_set: &'a DataSet,
        batch_size: usize,
        learning_rate: f64,
        activation_function: &'a Function,
    ) -> Map<Chunks<'_, DataVector>, impl FnMut(&'a [DataVector])> {
        data_set
            .training_data
            .chunks(batch_size)
            .map(move |data_slice: &'a [DataVector]| {
                self.step(
                    self.calculate_batch_step(data_slice, activation_function)
                        .unwrap(),
                    learning_rate / data_slice.len() as f64,
                )
            })
    }

    // train using stochastic gradient descent
    pub fn train_verbose(
        &mut self,
        data_set: &DataSet,
        batch_size: usize,
        learning_rate: f64,
        activation_function: &Function,
    ) {
        let map = self.train(data_set, batch_size, learning_rate, activation_function);

        let total_iterations = map.len() as f64;
        let mut current_iteration = 1.0;

        let mut loading_indicator: [char; 10] = ['_'; 10];

        map.for_each(|_| {
            current_iteration += 1.0;
            let fraction = (current_iteration - 1.0) / total_iterations;

            loading_indicator[(fraction * 9.0) as usize] = '█';

            print!(
                "\rTraining in progress: {} - {:0>3.2}% complete",
                loading_indicator.iter().collect::<String>(),
                fraction * 100.0,
            )
        });
        println!("");
    }

    fn calculate_batch_step(
        &self,
        data_set: &[DataVector],
        activation_function: &Function,
    ) -> Option<Self> {
        let mut delta_network = NeuralNetwork::zeros(self._shape.clone());
        for d in data_set.iter() {
            let training_data = d;
            let mut nodes = self.propagate_returning_all_nodes_prior_to_activation(
                &training_data.data,
                activation_function.activate,
            );

            let nodes_cur = nodes.pop()?;
            let mut index = nodes.len();

            let mut delta_cost_by_delta_nodes = NeuralNetwork::cost_derivative(
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

        let correct_filtered = data_set.testing_data.par_iter().filter(|image| {
            let res = self.propagate(&image.data, activation_function.activate);
            if res.index_of_max() == (image.label as usize) {
                return true;
            }

            false
        });

        let u = correct_filtered.collect::<Vec<&DataVector>>().len();
        u as f64 / data_set_length
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        calculus::functions::Function, linear_algebra::Matrix,
        machine_learning::neural_network::tests::init_network,
    };

    use super::NeuralNetwork;

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

        assert_eq!(x, y);
    }

    #[test]
    fn test_cost() {
        let (nn, ds) = init_network(vec![1, 2, 2]);
        let f = Function::sigmoid();

        let output = nn.propagate(&ds.testing_data.first().unwrap().data, f.activate);
        let cost = NeuralNetwork::_cost(&output, 1);

        assert!(cost[(0, 0)] - 0.3 < 0.01);
        assert!(cost[(1, 0)] - 0.2 < 0.01);
    }

    #[test]
    fn test_cost_derivative() {
        let (nn, ds) = init_network(vec![1, 2, 2]);
        let f = Function::sigmoid();

        let output = nn.propagate(&ds.testing_data.first().unwrap().data, f.activate);
        let derivative = NeuralNetwork::cost_derivative(&output, 1);

        assert!(derivative[(0, 0)] - 1.1 < 0.01);
        assert!(derivative[(1, 0)] + 0.9 < 0.01);
    }

    #[test]
    fn test_batch_step() {
        let (nn, ds) = init_network(vec![1, 2, 2]);
        let batch_step = nn.calculate_batch_step(&ds.training_data[0..1], &Function::sigmoid());

        match batch_step {
            Some(step) => {
                assert!(step._biases[0][(0, 0)] - 0.0012 < 0.0001);
                assert!(step._biases[0][(1, 0)] - 0.0012 < 0.0001);

                assert!(step._weigths[0][(0, 0)] == 0.0);
                assert!(step._weigths[0][(1, 0)] == 0.0);

                assert!(step._biases[1][(0, 0)] - 0.27 < 0.01);
                assert!(step._biases[1][(1, 0)] - 0.27 < 0.01);

                assert!(step._weigths[1][(0, 0)] - 0.14 < 0.01);
                assert!(step._weigths[1][(1, 0)] + 0.11 < 0.01);
                assert!(step._weigths[1][(0, 1)] - 0.14 < 0.01);
                assert!(step._weigths[1][(1, 1)] + 0.11 < 0.01);
            }
            None => panic!("No batch step calculated."),
        }
    }

    #[test]
    fn test_training_sigmoid() {
        let (mut nn, ds) = init_network(vec![1, 3, 2]);

        let f = Function::sigmoid();

        nn.train(&ds, 1, 1.0, &f).for_each(|_| {});

        assert!(
            0 == nn
                .propagate(&Matrix::from_vec(1, 1, vec![0.7]), f.activate)
                .index_of_max()
        );

        assert!(
            1 == nn
                .propagate(&Matrix::from_vec(1, 1, vec![0.4]), f.activate)
                .index_of_max()
        );
    }

    #[test]
    fn test_training_normal_arctan() {
        let (mut nn, ds) = init_network(vec![1, 3, 2]);

        let f = Function::normal_arctan();

        nn.train(&ds, 1, 1.0, &f).for_each(|_| {});

        assert!(
            0 == nn
                .propagate(&Matrix::from_vec(1, 1, vec![0.7]), f.activate)
                .index_of_max()
        );
        assert!(
            1 == nn
                .propagate(&Matrix::from_vec(1, 1, vec![0.4]), f.activate)
                .index_of_max()
        );
    }

    #[test]
    fn test_testing_network() {
        let (mut nn, ds) = init_network(vec![1, 3, 2]);
        let f = Function::normal_arctan();

        nn.train(&ds, 1, 1.0, &f).for_each(|_| {});

        assert!(nn.test(&ds, &f) > 0.7);
    }
}
