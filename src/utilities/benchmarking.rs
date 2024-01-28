use std::time::SystemTime;

pub fn _time_it<T>(func: impl FnOnce() -> T) -> f64 {
    let s = SystemTime::now();
    func();
    let e = SystemTime::now();
    let duration = e.duration_since(s).unwrap();
    duration.as_secs_f64()
}

#[cfg(test)]
mod test {
    use crate::{
        calculus::functions::Function,
        machine_learning::{
            dataset::{DataSet, DataVector},
            neural_network::{test_config::init_network, NeuralNetwork},
        },
    };

    use super::_time_it;

    impl NeuralNetwork {
        pub fn test_sync(&self, data_set: &DataSet, activation_function: &Function) -> f64 {
            let data_set_length = data_set.testing_data.len() as f64;

            let correct_filtered = data_set.testing_data.iter().filter(|image| {
                let res = self.propagate(&image.data, activation_function.activate);
                if res.index_of_max() == (image.label as usize) {
                    return true;
                }

                false
            });

            let u = correct_filtered.collect::<Vec<&DataVector>>().len();
            u as f64 / data_set_length
        }
    }

    #[test]
    fn benchmark_testing_network() {
        let (nn, ds) = init_network(vec![30, 20]);
        let a = _time_it(|| nn.test(&ds, &Function::sigmoid()));
        let b = _time_it(|| nn.test_sync(&ds, &Function::sigmoid()));
        println!("par: {a} - sync: {b}");
        assert!(a < b);

        let (nn, ds) = init_network(vec![300, 200]);
        let a = _time_it(|| nn.test(&ds, &Function::sigmoid()));
        let b = _time_it(|| nn.test_sync(&ds, &Function::sigmoid()));
        println!("par: {a} - sync: {b}");
        assert!(a < b);
    }
}
