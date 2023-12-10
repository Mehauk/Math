mod calculus;
mod machine_learning;
mod utilities;

use std::{error::Error, io::stdin};

use machine_learning::{
    dataset::{
        mnist::{parse_mnist, INPUT_SIZE},
        DataSet,
    },
    neural_network::NueralNetwork,
};

use crate::calculus::functions::Function;

// initialize constant values
const OUTPUT_SIZE: usize = 26;

fn main() -> Result<(), Box<dyn Error>> {
    // 28*28 is the expected image dimensions
    let ds = DataSet::load_data("src/assets/machine_learning/", "letters", parse_mnist)?;
    let mut nn = NueralNetwork::random(vec![INPUT_SIZE, 20, 20, OUTPUT_SIZE]);

    let sigmoid = Function::sigmoid();

    let mut input: String;
    loop {
        input = String::new();
        println!("1\t- reset the neural network");
        println!("2\t- train the network");
        println!("3\t- test the network");
        println!("4\t- exit");
        println!("");
        stdin().read_line(&mut input)?;
        println!("");
        let chosen = input.trim().parse::<i8>().unwrap_or(-1);

        match chosen {
            1 => {
                nn = NueralNetwork::random(vec![INPUT_SIZE, 16, 16, OUTPUT_SIZE]);
            }

            2 => {
                println!("Select a batch size (default 16): ");
                println!("");
                input = String::new();
                stdin().read_line(&mut input)?;
                let batch_size = input.trim().parse::<u32>().unwrap_or(16);
                nn.train(&ds, batch_size as usize, 1.0, &sigmoid);
                println!("");
            }

            3 => { 
                println!("Testing in Progress...");
                println!(
                    "Testing completed with {}% accuracy\n",
                    nn.test(&ds, &sigmoid) * 100.0
                );
            }

            4 => break,

            _ => println!("Invalid Selection!"),
        }
    }

    Ok(())
}
