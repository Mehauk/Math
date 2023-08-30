use nalgebra::SMatrix;

pub mod dataset;

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
struct NueralNetwork<const I: usize, const L: usize, const O: usize> {
    input_matrix: (SMatrix<f64, L, I>, SMatrix<f64, L, 1>),
    hidden_layer: Vec<(SMatrix<f64, L, L>, SMatrix<f64, L, 1>)>,
    output_matrix: (SMatrix<f64, O, L>, SMatrix<f64, O, 1>),
}

impl<const I: usize, const L: usize, const O: usize> NueralNetwork<I, L, O> {
    pub fn random(number_of_hidden_layers: usize) -> NueralNetwork<I, L, O> {
        let mut hidden_layer: Vec<(SMatrix<f64, L, L>, SMatrix<f64, L, 1>)> = vec![];
        hidden_layer.reserve(number_of_hidden_layers + 1);
        for _ in 0..number_of_hidden_layers {
            hidden_layer.push((
                SMatrix::<f64, L, L>::from_vec((0..L * L).map(|_| _random()).collect()),
                SMatrix::<f64, L, 1>::from_vec((0..L * 1).map(|_| _random()).collect()),
            ))
        }

        let input_matrix = (
            SMatrix::<f64, L, I>::from_vec((0..L * I).map(|_| _random()).collect()),
            SMatrix::<f64, L, 1>::from_vec((0..L * 1).map(|_| _random()).collect()),
        );

        let output_matrix = (
            SMatrix::<f64, O, L>::from_vec((0..O * L).map(|_| _random()).collect()),
            SMatrix::<f64, O, 1>::from_vec((0..O * 1).map(|_| _random()).collect()),
        );

        NueralNetwork {
            input_matrix,
            hidden_layer,
            output_matrix,
        }
    }

    pub fn propagate(&self, input: SMatrix<f64, I, 1>) -> SMatrix<f64, O, 1> {
        // construct the first layer of nodes to start the forward propagation.
        // LxI * Ix1 => Lx1
        let mut propagating_nodes = self.input_matrix.0 * input + self.input_matrix.1;
        propagating_nodes.apply(_arctan);

        // propagate through each layer in the hidden_layer
        for matrix in self.hidden_layer.iter() {
            // LxL * Lx1 => Lx1
            propagating_nodes = matrix.0 * propagating_nodes + matrix.1;
            propagating_nodes.apply(_arctan);
        }

        // calculate the resulting outputs
        // OxL * Lx1 => Ox1
        let mut output = self.output_matrix.0 * propagating_nodes + self.output_matrix.1;
        output.apply(_arctan);
        output
    }
}

fn _arctan(t: &mut f64) {
    *t = t.atan() / (std::f64::consts::PI / 2.0);
}

/// Return value between `[-1, 1]`
fn _random() -> f64 {
    rand::random::<f64>() * 2.0 - 1.0
}
