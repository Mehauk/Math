use nalgebra::SMatrix;

pub mod dataset;

/// Contruct a NeuralNetwork with;
/// - `I` number of inputs
/// - `L` number of nodes in each hidden_layer
/// - `O` number of output nodes.
///
/// ### parameters
/// - `imput_matrix` - A `IxL` matrix that holds the wights used in calculating the first layer of nodes from the input.
/// - `hidden_layer` - An arbitrary number of `LxL` weight matrices used in calculating nodes for the next layer.
/// - `imput_matrix` - A `OxL` matrix that holds the wights used in calculating the output nodes from the hidden layer.
struct NueralNetwork<const I: usize, const L: usize, const O: usize> {
    input_matrix: SMatrix<f64, L, I>,
    hidden_layer: Vec<SMatrix<f64, L, L>>,
    output_matrix: SMatrix<f64, O, L>,
}

impl<const I: usize, const L: usize, const O: usize> NueralNetwork<I, L, O> {
    pub fn pass_through(&self, input: SMatrix<f64, I, 1>) {
        // construct the first layer of nodes to start the forward propagation.
        // LxI * Ix1 => Lx1
        let mut propagating_nodes = self.input_matrix * input;

        // propagate through each layer in the hidden_layer
        for matrix in self.hidden_layer.iter() {
            // LxL * Lx1 => Lx1
            propagating_nodes = matrix * propagating_nodes;
        }

        // calculate the resulting outputs
        // OxL * Lx1 => Ox1
        let output = self.output_matrix * propagating_nodes;
    }
}
