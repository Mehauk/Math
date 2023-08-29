mod machine_learning;

use machine_learning::dataset::{parse_mnist, DataSet};

fn main() -> Result<(), std::io::Error> {
    // 28*28 is the expected image dimensions
    let _d = DataSet::<{ 28 * 28 }, 1>::load_data(
        "src/assets/machine_learning/",
        "letters",
        parse_mnist,
    )?;

    Ok(())
}
