use nalgebra::SMatrix;

use crate::calculus::functions::{sigmoid, sigmoid_derivative};

use super::dataset::DataSet;

/// Contruct a NeuralNetwork with;
/// - `I` number of inputs
/// - `L` number of nodes in each hidden_layer
/// - `O` number of output nodes.
///
/// ### Parameters
/// - `input_matrix : Tuple(SMatrix, Smatrix)`
///     - A `IxL` matrix that holds the wights used in calculating the first layer of nodes from the input.
///     - A `Lx1` matrix that holds the biases used in calculating the first layer of nodes from the input.
/// - `hidden_layer : Vec<Tuple(SMatrix, Smatrix)>`
///     - An arbitrary number of `LxL` weight matrices used in calculating nodes for the next layer.
///     - An arbitrary number of `Lx1` bias matrices used in calculating nodes for the next layer.
/// - `output_matrix : Tuple(SMatrix, Smatrix)`
///     - A `OxL` matrix that holds the wights used in calculating the output nodes from the hidden layer.
///     - A `Ox1` matrix that holds the biases used in calculating the output nodes from the hidden layer.
pub struct NueralNetwork<
    const INPUT_SIZE: usize,
    const NODES_PER_LAYER: usize,
    const LAYERS: usize,
    const OUTPUT_SIZE: usize,
> {
    _input_matrix: (
        SMatrix<f64, NODES_PER_LAYER, INPUT_SIZE>,
        SMatrix<f64, NODES_PER_LAYER, 1>,
    ),
    _hidden_layer: Vec<(
        SMatrix<f64, NODES_PER_LAYER, NODES_PER_LAYER>,
        SMatrix<f64, NODES_PER_LAYER, 1>,
    )>,
    _output_matrix: (
        SMatrix<f64, OUTPUT_SIZE, NODES_PER_LAYER>,
        SMatrix<f64, OUTPUT_SIZE, 1>,
    ),
}

impl<const I: usize, const N: usize, const L: usize, const O: usize> NueralNetwork<I, N, L, O> {
    pub fn random() -> NueralNetwork<I, N, L, O> {
        let input_matrix = (
            SMatrix::<f64, N, I>::new_random(),
            SMatrix::<f64, N, 1>::new_random(),
        );

        let hidden_layer: Vec<(SMatrix<f64, N, N>, SMatrix<f64, N, 1>)> = (0..L - 1)
            .map(|_| {
                (
                    SMatrix::<f64, N, N>::new_random(),
                    SMatrix::<f64, N, 1>::new_random(),
                )
            })
            .collect();

        let output_matrix = (
            SMatrix::<f64, O, N>::new_random(),
            SMatrix::<f64, O, 1>::new_random(),
        );

        NueralNetwork {
            _input_matrix: input_matrix,
            _hidden_layer: hidden_layer,
            _output_matrix: output_matrix,
        }
    }

    pub fn zeros() -> NueralNetwork<I, N, L, O> {
        let input_matrix = (SMatrix::<f64, N, I>::zeros(), SMatrix::<f64, N, 1>::zeros());

        let hidden_layer: Vec<(SMatrix<f64, N, N>, SMatrix<f64, N, 1>)> = (0..L - 1)
            .map(|_| (SMatrix::<f64, N, N>::zeros(), SMatrix::<f64, N, 1>::zeros()))
            .collect();

        let output_matrix = (SMatrix::<f64, O, N>::zeros(), SMatrix::<f64, O, 1>::zeros());

        NueralNetwork {
            _input_matrix: input_matrix,
            _hidden_layer: hidden_layer,
            _output_matrix: output_matrix,
        }
    }

    fn subtract(&mut self, other: Self) {
        self._input_matrix.0 -= other._input_matrix.0;
        self._input_matrix.1 -= other._input_matrix.1;

        for (a, b) in self._hidden_layer.iter_mut().zip(other._hidden_layer) {
            a.0 -= b.0;
            a.1 -= b.1;
        }

        self._output_matrix.0 -= other._output_matrix.0;
        self._output_matrix.1 -= other._output_matrix.1;
    }

    pub fn propagate(
        &self,
        input: &SMatrix<f64, I, 1>,
        activation_function: fn(&mut f64),
    ) -> SMatrix<f64, O, 1> {
        // construct the first layer of nodes to start the forward propagation.
        // LxI * Ix1 => Lx1
        let mut propagating_nodes = self._input_matrix.0 * input + self._input_matrix.1;
        propagating_nodes.apply(activation_function);

        // propagate through each layer in the hidden_layer
        for matrix in self._hidden_layer.iter() {
            // LxL * Lx1 => Lx1
            propagating_nodes = matrix.0 * propagating_nodes + matrix.1;
            propagating_nodes.apply(activation_function);
        }

        // calculate the resulting outputs
        // OxL * Lx1 => Ox1
        let mut output = self._output_matrix.0 * propagating_nodes + self._output_matrix.1;
        output.apply(activation_function);
        output
    }

    fn calculate_intermediate_nodes(
        &self,
        input: &SMatrix<f64, I, 1>,
        activation_function: fn(&mut f64),
    ) -> ([SMatrix<f64, N, 1>; L], SMatrix<f64, O, 1>) {
        // initialize resulting array;
        let mut nodes_array: [SMatrix<f64, N, 1>; L] = [SMatrix::zeros(); L];

        // construct the first layer of nodes to start the forward propagation.
        // LxI * Ix1 => Lx1
        nodes_array[0] = self._input_matrix.0 * input + self._input_matrix.1;
        nodes_array[0].apply(activation_function);

        // propagate through each layer in the hidden_layer
        let mut i: usize = 1;
        for matrix in self._hidden_layer.iter() {
            // LxL * Lx1 => Lx1
            nodes_array[i] = matrix.0 * nodes_array[i - 1] + matrix.1;
            nodes_array[i].apply(activation_function);
            i += 1;
        }

        // calculate the resulting outputs
        // OxL * Lx1 => Ox1
        let mut output = self._output_matrix.0 * nodes_array[i - 1] + self._output_matrix.1;
        output.apply(activation_function);
        (nodes_array, output)
    }

    /// calculates the cost the nueral network; `C = (R - E)^2`
    /// - `C` cost Matrix
    /// - `R` resulting output Matrix of network
    /// - `E` expected output Matrix contructed from label
    pub fn _cost(&result_matrix: &SMatrix<f64, O, 1>, label: u8) -> SMatrix<f64, O, 1> {
        let mut expected_matrix: SMatrix<f64, O, 1> = SMatrix::<f64, O, 1>::zeros();
        expected_matrix[(label as usize - 1, 0)] = 1.0;

        expected_matrix = result_matrix - expected_matrix;
        expected_matrix.apply(|x: &mut f64| *x = x.powi(2));
        expected_matrix
    }

    /// calculates the derivative of the cost; `C' = 2(R - E)
    /// - `C` cost Matrix
    /// - `R` resulting output Matrix of network
    /// - `E` expected output Matrix contructed from label
    pub fn _cost_derivative(&result_matrix: &SMatrix<f64, O, 1>, label: u8) -> SMatrix<f64, O, 1> {
        let mut expected_matrix: SMatrix<f64, O, 1> = SMatrix::<f64, O, 1>::zeros();
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
        println!(
            "\r{} - {:0>3.2}% complete",
            loading_indicator.iter().collect::<String>(),
            0.0
        );
        for n in 0..number_of_batches {
            let fraction = n as f64 / number_of_batches as f64;
            loading_indicator[(fraction * 9.0) as usize] = '█';
            self.calculate_and_apply_batch_step(data_set, batch_size);

            print!(
                "\r{} - {:0>3.2}% complete",
                loading_indicator.iter().collect::<String>(),
                fraction * 100.0,
            );
        }
        self.calculate_and_apply_batch_step(data_set, remaining_data);

        loading_indicator[9] = '█';
        print!(
            "\r{} - {:0>3.2}% complete",
            loading_indicator.iter().collect::<String>(),
            100.0,
        );
    }

    fn calculate_and_apply_batch_step(&mut self, data_set: &DataSet<I>, batch_size: usize) {
        let mut delta_network = NueralNetwork::<I, N, L, O>::zeros();
        for i in 0..batch_size {
            let input = &data_set.training_data[i].pixels;
            let label = &data_set.training_data[i].label;
            let (nodes, output_nodes) = self.calculate_intermediate_nodes(input, sigmoid);

            // calculate output bias changes
            delta_network._output_matrix.1 += Self::_cost_derivative(&output_nodes, *label)
                .component_mul(&output_nodes.apply_into(sigmoid_derivative));

            // calculate output weights changes
            delta_network._output_matrix.0 +=
                delta_network._output_matrix.1 * nodes[L - 1].transpose();

            // calculate delta for previous nodes
            let mut delta_intermediate_nodes: SMatrix<f64, N, 1> =
                self._output_matrix.0.transpose() * delta_network._output_matrix.1;

            for ix in 0..L - 1 {
                let mut delta_layer = delta_network._hidden_layer[L - 2 - ix];

                // calculate biases changes for selected layer
                delta_layer.1 += delta_intermediate_nodes
                    .component_mul(&nodes[L - 1 - ix].apply_into(sigmoid_derivative));

                // calculate weight changes for selected layer
                delta_layer.0 += delta_layer.1 * nodes[L - 2 - ix].transpose();

                // calculate delta for previous nodes
                delta_intermediate_nodes =
                    self._hidden_layer[L - 2 - ix].0.transpose() * delta_layer.1;
            }

            // calculate input bias changes
            delta_network._input_matrix.1 +=
                delta_intermediate_nodes.component_mul(&nodes[0].apply_into(sigmoid_derivative));

            // calculate input weights changes
            delta_network._input_matrix.0 += delta_network._input_matrix.1 * input.transpose();
        }
        self.subtract(delta_network);
    }

    pub fn test(&self, data_set: &DataSet<I>) -> f64 {
        let data_set_length = data_set.testing_data.len() as f64;
        let mut total_correct = 0.0;

        for image in data_set.testing_data.iter() {
            let res = self.propagate(&image.pixels, sigmoid);
            if res.argmax().0 == (image.label as usize - 1) {
                total_correct += 1.0;
            }
        }

        total_correct / data_set_length
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::SMatrix;

    use crate::{
        calculus::functions::sigmoid,
        machine_learning::dataset::{DataSet, ImageData},
    };

    use super::NueralNetwork;

    impl<const I: usize, const N: usize, const L: usize, const O: usize> NueralNetwork<I, N, L, O> {
        fn display_nodes(&self, input: &SMatrix<f64, I, 1>) {
            let mut string: String = format!("{:.2?} > ", input);

            let (_, output) = self.calculate_intermediate_nodes(input, sigmoid);

            // for n in nodes {
            //     string += &format!("{:.2} - ", n);
            // }

            string += &format!("{:.2?}", output);

            println!("{:.2?}", string);
        }

        fn test_debug(&self, data_set: &DataSet<I>) -> f64 {
            let data_set_length = data_set.testing_data.len() as f64;
            let mut total_correct = 0.0;

            println!("Testing:\n");

            for image in data_set.testing_data.iter() {
                let res = self.propagate(&image.pixels, sigmoid);
                print!("{:.2?}>{:.2?} ||| ", image.pixels, res);
                if res.argmax().0 == (image.label as usize - 1) {
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
                    ImageData::<1>::new(SMatrix::from_vec(vec![y]), if y < 0.5 { 1 } else { 2 })
                })
                .collect(),
            testing_data: (0..100)
                .map(|x: usize| {
                    let y: f64 = x as f64 / 100.0;
                    ImageData::<1>::new(SMatrix::from_vec(vec![y]), if y < 0.5 { 1 } else { 2 })
                })
                .collect(),
        };

        print!("\nTesting in Progress\n");
        println!(
            "Testing completed with {}% accuracy\n",
            nn.test_debug(&ds) * 100.0
        );
        nn.display_nodes(&ds.testing_data[1].pixels);
        // nn.display_nodes(&ds.testing_data[11].pixels);
        nn.display_nodes(&ds.testing_data[88].pixels);
        println!("---");
        nn.train(&ds, 5);
        println!("\n---");
        print!("\nTesting in Progress\n");
        println!(
            "Testing completed with {}% accuracy\n",
            nn.test_debug(&ds) * 100.0
        );
        nn.display_nodes(&ds.testing_data[1].pixels);
        // nn.display_nodes(&ds.testing_data[11].pixels);
        nn.display_nodes(&ds.testing_data[88].pixels);
    }
}
