mod calculus;
mod machine_learning;
mod utilities;

use std::io::stdin;

use machine_learning::{
    dataset::{
        mnist::{parse_mnist, INPUT_SIZE},
        DataSet,
    },
    neural_network::NueralNetwork,
};

// initialize constant values
const LAYERS_SIZE: usize = 8;
const OUTPUT_SIZE: usize = 26;

fn main() -> Result<(), std::io::Error> {
    // 28*28 is the expected image dimensions
    let ds =
        DataSet::<INPUT_SIZE>::load_data("src/assets/machine_learning/", "letters", parse_mnist)?;

    let mut input = String::new();

    loop {
        input = String::new();
        println!("1\t- reset the neural network");
        println!("2\t- train the network");
        println!("3\t- test the network");
        println!("4\t- exit");
        println!("");
        stdin().read_line(&mut input)?;

        let mut nn = NueralNetwork::<INPUT_SIZE, LAYERS_SIZE, 2, OUTPUT_SIZE>::random();

        print!("\nTesting in Progress\n");
        println!(
            "Testing completed with {}% accuracy\n",
            nn.test(&ds) * 100.0
        );
        println!("---");
        nn.train(&ds, 100);
        println!("\n---");
        print!("\nTesting in Progress\n");
        println!(
            "Testing completed with {}% accuracy\n",
            nn.test(&ds) * 100.0
        );
    }

    Ok(())
}
