pub mod dataset;

/// Contruct a NeuralNetwork with `L` number of layers
/// Each layer can have up to 255 (u8) nodes
///
/// ### parameters
/// - `dimensions: [u8, L]` - an u8 array of len L
struct _NueralNetwrok<const L: usize> {
    dimensions: [u8; L],
}
