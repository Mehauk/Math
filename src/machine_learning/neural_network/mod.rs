use nalgebra::{DMatrix, DVector};

use crate::calculus::functions::{sigmoid, sigmoid_derivative};

use super::dataset::DataSet;

/// ### Parameters
/// - `_weights` : `Vec<DMatrix<f64>>`
/// - `hidden_layer` : `Vec<DVector<f64>>`
pub struct NueralNetwork {
    _weigths: Vec<DMatrix<f64>>,
    _biases: Vec<DVector<f64>>,
}

impl NueralNetwork {
    pub fn random(shape: [usize]) -> NueralNetwork {
        let length = shape.len();
        let weigths: Vec<DMatrix<f64>> = shape[1..length]
            .iter()
            .zip(shape[0..length - 1])
            .map(|a, b| random_matrix(a, b));
        let biases: Vec<DVector<f64>> = shape[1..length].iter().map(|a| random_matrix(a, 1));

        NueralNetwork {
            _weigths: weigths,
            _biases: biases,
        }
    }

    pub fn zeros(shape: [usize]) -> NueralNetwork {
        let length = shape.len();
        let weigths: Vec<DMatrix<f64>> = shape[1..length]
            .iter()
            .zip(shape[0..length - 1])
            .map(|a, b| DMatrix::<f64>::zeros(a, b));
        let biases: Vec<DVector<f64>> =
            shape[1..length].iter().map(|a| DMatrix::<f64>::zeros(a, 1));

        NueralNetwork {
            _weigths: weigths,
            _biases: biases,
        }
    }

    fn _step(&mut self, other: Self, learning_rate: f64) {
        for (a, b) in self._weigths.iter_mut().zip(other._weigths) {
            a.0 -= b.0 * learning_rate;
            a.1 -= b.1 * learning_rate;
        }

        for (a, b) in self._biases.iter_mut().zip(other._biases) {
            a.0 -= b.0 * learning_rate;
            a.1 -= b.1 * learning_rate;
        }
    }

    pub fn propagate(
        &self,
        input: &DVector<f64>,
        activation_function: fn(&mut f64),
    ) -> DVector<f64> {

        let mut propagating_nodes: DVector<f64> = input;

        for (weight_matrix, bias_vector) in self._weigths.iter().zip(self._biases) {
            propagating_nodes = weight_matrix * input + bias_vector;
        }

        propagating_nodes
    }

    fn calculate_intermediate_nodes(
        &self,
        input: &DMatrix<f64>,
        activation_function: fn(&mut f64),
    ) -> Vec<DVector<f64>> {
        // initialize resulting array;
        let mut nodes_array: Vec<DVector<f64>>;

        for _ in 0..L {
            nodes_array.push(DMatrix::zeros(N, 1))
        }
    }

    /// calculates the cost the nueral network; `C = (R - E)^2`
    /// - `C` cost Matrix
    /// - `R` resulting output Matrix of network
    /// - `E` expected output Matrix contructed from label
    pub fn _cost(result_matrix: &DMatrix<f64>, label: u8) -> DMatrix<f64> {
        let mut expected_matrix: DMatrix<f64> = DMatrix::<f64>::zeros(O, 1);
        expected_matrix[(label as usize - 1, 0)] = 1.0;

        expected_matrix = result_matrix - expected_matrix;
        expected_matrix.apply(|x: &mut f64| *x = x.powi(2));
        expected_matrix
    }

    /// calculates the derivative of the cost; `C' = 2(R - E)
    /// - `C` cost Matrix
    /// - `R` resulting output Matrix of network
    /// - `E` expected output Matrix contructed from label
    pub fn _cost_derivative(result_matrix: &DMatrix<f64>, label: u8) -> DMatrix<f64> {
        let mut expected_matrix: DMatrix<f64> = DMatrix::<f64>::zeros(O, 1);
        expected_matrix[(label as usize - 1, 0)] = 1.0;

        expected_matrix = result_matrix - expected_matrix;
        expected_matrix.scale(2.0)
    }

    // train using stochastic gradient descent
    pub fn train(&mut self, data_set: &DataSet<I>, batch_size: usize) {
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

    fn calculate_and_apply_batch_step(&mut self, data_set: &DataSet<I>, batch_size: usize) {
        let mut delta_network = NueralNetwork::<I, N, L, O>::zeros();
        for i in 0..batch_size {
            let input = &data_set.training_data[i].pixels;
            let label = &data_set.training_data[i].label;
            let (mut nodes, output_nodes) = self.calculate_intermediate_nodes(input, sigmoid);

            // calculate output bias changes
            delta_network._output_matrix.1 += Self::_cost_derivative(&output_nodes, *label)
                .component_mul(&output_nodes.apply_into(sigmoid_derivative));

            // calculate output weights changes
            delta_network._output_matrix.0 +=
                &delta_network._output_matrix.1 * nodes[L - 1].transpose();

            // calculate delta for previous nodes
            let mut delta_intermediate_nodes: DMatrix<f64> =
                self._output_matrix.0.transpose() * &delta_network._output_matrix.1;

            for ix in 0..L - 1 {
                let delta_layer = &mut delta_network._hidden_layer[L - 2 - ix];

                // calculate biases changes for selected layer
                nodes[L - 1 - ix].apply(sigmoid_derivative);
                delta_layer.1 += delta_intermediate_nodes.component_mul(&nodes[0]);

                // calculate weight changes for selected layer
                delta_layer.0 += &delta_layer.1 * nodes[L - 2 - ix].transpose();

                // calculate delta for previous nodes
                delta_intermediate_nodes =
                    self._hidden_layer[L - 2 - ix].0.transpose() * &delta_layer.1;
            }

            // calculate input bias changes
            nodes[0].apply(sigmoid_derivative);
            delta_network_inputt_matrix.1 += delta_intermediate_nodes.component_mul(&nodes[0]);

            // calculate input weights changes
            delta_network_inputt_matrix.0 += &delta_network_inputt_matrix.1 * input.transpose();
        }
        self._step(delta_network, 0.1 / batch_size as f64);
    }

    pub fn test(&self, data_set: &DataSet<I>) -> f64 {
        let data_set_length = data_set.testing_data.len() as f64;
        let mut total_correct = 0.0;

        for image in data_set.testing_data.iter() {
            let res = self.propagate(&image.pixels, sigmoid);
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

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use crate::{
        calculus::functions::sigmoid,
        machine_learning::dataset::{DataSet, ImageData},
    };

    use super::NueralNetwork;

    impl<const I: usize, const N: usize, const L: usize, const O: usize> NueralNetwork<I, N, L, O> {
        fn _display_nodes(&self, input: &DMatrix<f64>) {
            let mut string: String = format!("{:.2?} > ", input);

            let (_, output) = self.calculate_intermediate_nodes(input, sigmoid);

            // for n in nodes {
            //     string += &format!("{:.2} - ", n);
            // }

            string += &format!("{:.2?}", output);

            println!("{:.2?}", string);
        }

        fn _test_debug(&self, data_set: &DataSet<I>) -> f64 {
            let data_set_length = data_set.testing_data.len() as f64;
            let mut total_correct = 0.0;

            println!("Testing:\n");

            for image in data_set.testing_data.iter() {
                let res = self.propagate(&image.pixels, sigmoid);
                // print!("{:.2?}>{:.2?} ||| ", image.pixels, res);
                if res.column(0).argmax().0 == (image.label as usize - 1) {
                    total_correct += 1.0;
                }
            }

            total_correct / data_set_length
        }
    }

    #[test]
    fn test_network_1_2_2_2() {
        let mut nn = NueralNetwork::<1, 1, 2, 2>::random();
        let ds = DataSet::<1> {
            training_data: (0..10000)
                .map(|_| {
                    let y: f64 = rand::random();
                    ImageData::<1>::_new(
                        DMatrix::from_vec(1, 1, vec![y]),
                        if y < 0.5 { 1 } else { 2 },
                    )
                })
                .collect(),
            testing_data: (0..100)
                .map(|x: usize| {
                    let y: f64 = x as f64 / 100.0;
                    ImageData::<1>::_new(
                        DMatrix::from_vec(1, 1, vec![y]),
                        if y < 0.5 { 1 } else { 2 },
                    )
                })
                .collect(),
        };

        print!("\nTesting in Progress\n");
        println!(
            "Testing completed with {}% accuracy\n",
            nn._test_debug(&ds) * 100.0
        );
        // nn._display_nodes(&ds.testing_data[1].pixels);
        // nn._display_nodes(&ds.testing_data[11].pixels);
        // nn._display_nodes(&ds.testing_data[88].pixels);
        println!("---");
        nn.train(&ds, 16);
        println!("\n---");
        print!("\nTesting in Progress\n");
        println!(
            "Testing completed with {}% accuracy\n",
            nn._test_debug(&ds) * 100.0
        );
        // nn._display_nodes(&ds.testing_data[1].pixels);
        // nn._display_nodes(&ds.testing_data[11].pixels);
        // nn._display_nodes(&ds.testing_data[88].pixels);
    }
}
