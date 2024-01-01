mod calculus;
mod machine_learning;
mod utilities;

use std::{io::stdin, io::Error};

use machine_learning::{
    dataset::{
        mnist::{parse_mnist, INPUT_SIZE},
        DataSet,
    },
    neural_network::NeuralNetwork,
};

use crate::calculus::functions::Function;

// initialize constant values
const OUTPUT_SIZE: usize = 26;

fn main() -> Result<(), Error> {
    // 28*28 is the expected image dimensions
    let mut ds = DataSet::load_data("src/assets/machine_learning/", "letters", parse_mnist)?;
    let mut nn = create_nn();

    let mut activation_function = Function::sigmoid();
    choose_activation_function(&mut activation_function);

    let mut input: String;
    loop {
        input = String::new();
        println!("0\t- load the last untrained network");
        println!("1\t- reset the neural network");
        println!("2\t- train the network");
        println!("3\t- test the network");
        println!("4\t- exit");
        println!("");
        stdin().read_line(&mut input)?;
        println!("");
        let chosen = input.trim().parse::<i8>().unwrap_or(-1);

        match chosen {
            0 => {
                nn = NeuralNetwork::load("output/random_network.nn").unwrap_or_else(|_| create_nn())
            }

            1 => {
                nn = create_nn();
                choose_activation_function(&mut activation_function);
            }

            2 => {
                println!("Select the number of epochs to train for (default 30):");
                input = String::new();
                stdin().read_line(&mut input)?;
                println!("");
                let epochs = input.trim().parse::<u32>().unwrap_or(30);

                println!("Select a batch size (default 10):");
                input = String::new();
                stdin().read_line(&mut input)?;
                println!("");
                let batch_size = input.trim().parse::<u32>().unwrap_or(16);

                println!("Select a learning rate (default 1.0):");
                input = String::new();
                stdin().read_line(&mut input)?;
                let learning_rate = input.trim().parse::<f64>().unwrap_or(1.0);

                for epi in 0..epochs {
                    // randomize training data
                    ds.training_data
                        .sort_by(|_, _| rand::random::<f32>().partial_cmp(&0.5).unwrap());

                    println!("training {}", epi);
                    nn.train(
                        &ds,
                        batch_size as usize,
                        learning_rate,
                        &activation_function,
                    );

                    print!("Testing in Progress...");
                    print!(
                        "\rTesting completed with {}% accuracy\n",
                        nn.test(&ds, &activation_function) * 100.0
                    );
                }
                println!("");
            }

            3 => {
                print!("Testing in Progress...");
                print!(
                    "\rTesting completed with {}% accuracy\n",
                    nn.test(&ds, &activation_function) * 100.0
                );
            }

            4 => break,

            _ => println!("Invalid Selection!"),
        }
    }

    Ok(())
}

fn choose_activation_function(activation_function: &mut Function) {
    let mut input = String::new();
    println!("Choose an activation function (default sigmoid):");
    println!("1\t- sigmoid");
    println!("2\t- arctan");
    println!("3\t- normal_arctan");
    println!("4\t- relu");
    println!("5\t- leaky_relu");
    println!("");
    stdin().read_line(&mut input).unwrap_or_default();
    println!("");
    let function_choice = input.trim().parse::<u32>().unwrap_or(16);
    match function_choice {
        1 => *activation_function = Function::sigmoid(),
        2 => *activation_function = Function::arctan(),
        3 => *activation_function = Function::normal_arctan(),
        4 => *activation_function = Function::relu(),
        5 => *activation_function = Function::leaky_relu(),
        _ => *activation_function = Function::sigmoid(),
    }
}

fn create_nn() -> NeuralNetwork {
    let mut v = vec![INPUT_SIZE];

    let mut input = String::new();
    println!("Define the shape of the network with space separated values:");
    stdin().read_line(&mut input).unwrap_or_default();
    println!("");
    let mut iter = input.trim().split_whitespace();

    while let Some(cur) = iter.next() {
        v.push(cur.parse::<usize>().unwrap_or(10))
    }

    v.push(OUTPUT_SIZE);
    let n = NeuralNetwork::random(v);

    let _ = n.save("output/random_network.nn");

    n
}
