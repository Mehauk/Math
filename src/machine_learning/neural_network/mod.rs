use nalgebra::{DMatrix, DVector};

use crate::calculus::functions::{sigmoid, sigmoid_derivative};

use super::dataset::DataSet;

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

    fn _step(&mut self, other: Self, learning_rate: f64) {
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
    fn propagate_returning_all_nodes(
        &self,
        input: &DVector<f64>,
        activation_function: fn(&mut f64),
    ) -> Vec<DVector<f64>> {
        // initialize resulting array;
        let mut nodes_array: Vec<DVector<f64>> = vec![];

        let mut propagating_nodes: &DVector<f64> = input;

        for (weight_matrix, bias_vector) in self._weigths.iter().zip(&self._biases) {
            nodes_array.push(
                (weight_matrix * propagating_nodes + bias_vector).apply_into(activation_function),
            );
            propagating_nodes = nodes_array.last().unwrap();
        }

        nodes_array
    }

    /// calculates the cost the nueral network; `C = (R - E)^2`
    /// - `C` cost Matrix
    /// - `R` resulting output Matrix of network
    /// - `E` expected output Matrix contructed from label
    pub fn _cost(result_matrix: &DVector<f64>, label: u8) -> DVector<f64> {
        let mut m = result_matrix.clone();
        m[(label as usize - 1, 0)] -= 1.0;
        m.apply_into(|x| *x *= *x)
    }

    /// calculates the derivative of the cost; `C' = 2(R - E)
    /// - `C` cost Matrix
    /// - `R` resulting output Matrix of network
    /// - `E` expected output Matrix contructed from label
    pub fn _cost_derivative(result_matrix: &DVector<f64>, label: u8) -> DVector<f64> {
        let mut m = result_matrix.clone();
        m[(label as usize - 1, 0)] -= 1.0;
        m.scale(2.0)
    }

    // train using stochastic gradient descent
    pub fn train(&mut self, data_set: &DataSet, batch_size: usize) {
        println!("Training has commenced.");
        let data_set_length = data_set.training_data.len();
        let number_of_batches = data_set_length / batch_size;
        let remaining_data = data_set_length % batch_size;

        let mut loading_indicator: [char; 10] = ['_'; 10];
        println!("");
        print!(
            "\rTraining in progress: {} - {:0>3.2}% complete",
            loading_indicator.iter().collect::<String>(),
            0.0
        );
        for n in 0..number_of_batches {
            let fraction = n as f64 / number_of_batches as f64;
            loading_indicator[(fraction * 9.0) as usize] = '█';
            // self.calculate_and_apply_batch_step(data_set, batch_size);

            print!(
                "\rTraining in progress: {} - {:0>3.2}% complete",
                loading_indicator.iter().collect::<String>(),
                fraction * 100.0,
            );
        }
        // self.calculate_and_apply_batch_step(data_set, remaining_data);

        loading_indicator[9] = '█';
        print!(
            "\rTraining in progress: {} - {:0>3.2}% complete\n",
            loading_indicator.iter().collect::<String>(),
            100.0,
        );
    }

    fn calculate_batch_step(&mut self, data_set: &DataSet, batch_size: usize) {
        // todo: fix this and add activation/derivative abstraction
        let mut delta_network = NueralNetwork::zeros(self._shape.clone());
        for i in 0..batch_size {
            let training_data = &data_set.training_data[i];
            let nodes = self.propagate_returning_all_nodes(&training_data.data, sigmoid);
            let network_length = nodes.len();

            let delta_cost_by_delta_nodes =
                NueralNetwork::_cost_derivative(&nodes[network_length - 1], training_data.label);
            let delta_cost_by_delta_activation = delta_cost_by_delta_nodes.component_mul(
                &delta_cost_by_delta_nodes
                    .clone()
                    .apply_into(sigmoid_derivative),
            );
            let delta_cost_by_delta_biases = delta_cost_by_delta_activation;

            println!(
                "{:#?}",
                &delta_cost_by_delta_nodes
                    .clone()
                    .apply_into(sigmoid_derivative)
            );
            println!("{:#?}", delta_cost_by_delta_biases);
        }
        self._step(delta_network, 0.1 / batch_size as f64);
    }

    pub fn test(&self, data_set: &DataSet) -> f64 {
        let data_set_length = data_set.testing_data.len() as f64;
        let mut total_correct = 0.0;

        for image in data_set.testing_data.iter() {
            let res = self.propagate(&image.data, sigmoid);
            if res.column(0).argmax().0 == (image.label as usize - 1) {
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
        calculus::functions::sigmoid,
        machine_learning::dataset::{DataSet, DataVector},
    };

    use super::NueralNetwork;

    fn init_network() -> (NueralNetwork, DataSet) {
        let mut nn = NueralNetwork::zeros(vec![1, 2, 2]);
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
                    DataVector::_new(DVector::from_vec(vec![y]), if y < 0.5 { 1 } else { 2 })
                })
                .collect(),
            testing_data: (0..100)
                .map(|x: usize| {
                    let y: f64 = x as f64 / 100.0;
                    DataVector::_new(DVector::from_vec(vec![y]), if y < 0.5 { 1 } else { 2 })
                })
                .collect(),
        };

        return (nn, ds);
    }

    #[test]
    fn test_both_propagation_methods_are_equivalent() {
        let (nn, ds) = init_network();

        let x = nn
            .propagate_returning_all_nodes(&ds.training_data.first().unwrap().data, sigmoid)
            .last()
            .unwrap()
            .clone();
        let y = nn.propagate(&ds.training_data.first().unwrap().data, sigmoid);

        assert_eq!(x, y);
    }

    #[test]
    fn test_multiple_propagation_calls() {
        let (nn, ds) = init_network();

        let y = nn.propagate(&ds.training_data.first().unwrap().data, sigmoid);
        let x = nn.propagate(&ds.training_data.first().unwrap().data, sigmoid);

        assert_eq!(x, y);
    }

    #[test]
    fn test_cost() {
        let (nn, ds) = init_network();

        let output = nn.propagate(&ds.testing_data.first().unwrap().data, sigmoid);

        println!("{:#?}", NueralNetwork::_cost(&output, 1));
    }

    #[test]
    fn test_cost_derivative() {
        let (nn, ds) = init_network();

        let output = nn.propagate(&ds.testing_data.first().unwrap().data, sigmoid);
        let derivative = NueralNetwork::_cost_derivative(&output, 1);

        assert!(derivative[0] - 1.1 < 0.01);
        assert!(derivative[1] + 0.9 < 0.01);
    }

    #[test]
    fn test_fake() {
        let (mut nn, ds) = init_network();
        nn.calculate_batch_step(&ds, 1);
    }
}
