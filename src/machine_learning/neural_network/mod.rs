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
    pub fn _cost(mut result_matrix: DMatrix<f64>, label: u8) -> DMatrix<f64> {
        result_matrix[(label as usize - 1, 0)] -= 1.0;
        result_matrix.apply_into(|x| {
            *x *= *x;
        })
    }

    /// calculates the derivative of the cost; `C' = 2(R - E)
    /// - `C` cost Matrix
    /// - `R` resulting output Matrix of network
    /// - `E` expected output Matrix contructed from label
    pub fn _cost_derivative(mut result_matrix: DMatrix<f64>, label: u8) -> DMatrix<f64> {
        result_matrix[(label as usize - 1, 0)] -= 1.0;
        result_matrix.scale(2.0)
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
            self.calculate_and_apply_batch_step(data_set, batch_size);

            print!(
                "\rTraining in progress: {} - {:0>3.2}% complete",
                loading_indicator.iter().collect::<String>(),
                fraction * 100.0,
            );
        }
        self.calculate_and_apply_batch_step(data_set, remaining_data);

        loading_indicator[9] = '█';
        print!(
            "\rTraining in progress: {} - {:0>3.2}% complete\n",
            loading_indicator.iter().collect::<String>(),
            100.0,
        );
    }

    fn calculate_and_apply_batch_step(&mut self, data_set: &DataSet, batch_size: usize) {
        // let mut delta_network = NueralNetwork::zeros(self._shape);
        // for i in 0..batch_size {
        //     let input = &data_set.training_data[i].data;
        //     let label = &data_set.training_data[i].label;
        //     let mut nodes = self.propagate_returning_all_nodes(input, sigmoid);

        //     // change all below
        //     // calculate output bias changes
        //     delta_network._output_matrix.1 += Self::_cost_derivative(&output_nodes, *label)
        //         .component_mul(&output_nodes.apply_into(sigmoid_derivative));

        //     // calculate output weights changes
        //     delta_network._output_matrix.0 +=
        //         &delta_network._output_matrix.1 * nodes[L - 1].transpose();

        //     // calculate delta for previous nodes
        //     let mut delta_intermediate_nodes: DMatrix<f64> =
        //         self._output_matrix.0.transpose() * &delta_network._output_matrix.1;

        //     for ix in 0..L - 1 {
        //         let delta_layer = &mut delta_network._hidden_layer[L - 2 - ix];

        //         // calculate biases changes for selected layer
        //         nodes[L - 1 - ix].apply(sigmoid_derivative);
        //         delta_layer.1 += delta_intermediate_nodes.component_mul(&nodes[0]);

        //         // calculate weight changes for selected layer
        //         delta_layer.0 += &delta_layer.1 * nodes[L - 2 - ix].transpose();

        //         // calculate delta for previous nodes
        //         delta_intermediate_nodes =
        //             self._hidden_layer[L - 2 - ix].0.transpose() * &delta_layer.1;
        //     }

        //     // calculate input bias changes
        //     nodes[0].apply(sigmoid_derivative);
        //     delta_network_inputt_matrix.1 += delta_intermediate_nodes.component_mul(&nodes[0]);

        //     // calculate input weights changes
        //     delta_network_inputt_matrix.0 += &delta_network_inputt_matrix.1 * input.transpose();
        // }
        // self._step(delta_network, 0.1 / batch_size as f64);
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
                .map(|_| {
                    let y: f64 = rand::random();
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
}
