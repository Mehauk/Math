mod machine_learning;
mod utilities;

use machine_learning::dataset::{parse_mnist, DataSet};
use utilities::benchmarking::_time_it;

fn main() -> Result<(), std::io::Error> {
    // 28*28 is the expected image dimensions

    _time_it(|| -> Result<DataSet<{ 28 * 28 }, 1>, std::io::Error> {
        let d = DataSet::<{ 28 * 28 }, 1>::load_data(
            "src/assets/machine_learning/",
            "letters",
            parse_mnist,
        )?;

        Ok(d)
    });

    Ok(())
}
