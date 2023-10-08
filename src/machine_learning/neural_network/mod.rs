
use nalgebra::SMatrix;

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
pub struct NueralNetwork<const I: usize, const L: usize, const O: usize> {
    _input_matrix: (SMatrix<f64, L, I>, SMatrix<f64, L, 1>),
    _hidden_layer: Vec<(SMatrix<f64, L, L>, SMatrix<f64, L, 1>)>,
    _output_matrix: (SMatrix<f64, O, L>, SMatrix<f64, O, 1>),
}

impl<const I: usize, const L: usize, const O: usize> NueralNetwork<I, L, O> {
    pub fn random(number_of_layers: usize) -> NueralNetwork<I, L, O> {
        let input_matrix = (
            SMatrix::<f64, L, I>::from_vec((0..L * I).map(|_| _random()).collect()),
            SMatrix::<f64, L, 1>::from_vec((0..L * 1).map(|_| _random()).collect()),
        );

        let hidden_layer: Vec<(SMatrix<f64, L, L>, SMatrix<f64, L, 1>)> = (0..number_of_layers)
            .map(|_| {
                (
                    SMatrix::<f64, L, L>::from_vec((0..L * L).map(|_| _random()).collect()),
                    SMatrix::<f64, L, 1>::from_vec((0..L * 1).map(|_| _random()).collect()),
                )
            })
            .collect();

        let output_matrix = (
            SMatrix::<f64, O, L>::from_vec((0..O * L).map(|_| _random()).collect()),
            SMatrix::<f64, O, 1>::from_vec((0..O * 1).map(|_| _random()).collect()),
        );

        NueralNetwork {
            _input_matrix: input_matrix,
            _hidden_layer: hidden_layer,
            _output_matrix: output_matrix,
        }
    }

    pub fn _propagate(
        &self,
        input: SMatrix<f64, I, 1>,
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

    /// calculates the cost the nueral network; `C = (R - E)^2`
    /// - `C` cost Matrix
    /// - `R` resulting output Matrix of network
    /// - `E` expected output Matrix contructed from label
    pub fn _cost(&result_matrix: &SMatrix<f64, O, 1>, label: u8) -> SMatrix<f64, O, 1> {
        let mut expected_matrix: SMatrix<f64, O, 1> = SMatrix::<f64, O, 1>::zeros();
        expected_matrix[(label as usize, 0)] = 1.0;

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
        expected_matrix[(label as usize, 0)] = 1.0;

        expected_matrix = result_matrix - expected_matrix;
        expected_matrix.scale(2.0)
    }

    pub fn train(&self, data_set: &DataSet<I>) {}

    pub fn test(&self, data_set: &DataSet<I>) {}
}

/// Return value between `[-1, 1]`
fn _random() -> f64 {
    rand::random::<f64>() * 2.0 - 1.0
}
