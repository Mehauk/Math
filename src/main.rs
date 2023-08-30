mod machine_learning;
mod utilities;

use machine_learning::{
    dataset::{
        mnist::{parse_mnist, INPUT_SIZE},
        DataSet,
    },
    NueralNetwork,
};

// initialize constant values
const LAYERS_SIZE: usize = 16;
const OUTPUT_SIZE: usize = 26;

fn main() -> Result<(), std::io::Error> {
    // 28*28 is the expected image dimensions
    DataSet::<INPUT_SIZE>::load_data("src/assets/machine_learning/", "letters", parse_mnist)?;
    NueralNetwork::<INPUT_SIZE, LAYERS_SIZE, OUTPUT_SIZE>::random(2);
    Ok(())
}
