use std::io::Error;

use crate::linear_algebra::Matrix;

pub mod mnist;

pub struct DataSet {
    pub training_data: Vec<DataVector>,
    pub testing_data: Vec<DataVector>,
}

impl DataSet {
    pub fn load_data(
        dataset_path: &str,
        dataset_name: &str,
        parse_data: impl Fn(&str, &str, &str) -> Result<Vec<DataVector>, Error>,
    ) -> Result<DataSet, Error> {
        let _test_data = parse_data(dataset_path, dataset_name, "test")?;
        let _train_data = parse_data(dataset_path, dataset_name, "train")?;

        Ok(DataSet {
            testing_data: _test_data,
            training_data: _train_data,
        })
    }
}

#[derive(Debug)]
pub struct DataVector {
    /// `ArrayStorage` requires constant usize for Row and Column.
    ///
    /// We could also use `VecStorage` to create a matrix with
    /// dimensions calculated at runtime, but operations would be slower.
    ///
    /// values in the matrix are between 0-1 (inc), normalized from u8
    pub data: Matrix,
    pub label: u8,
}

/// Contructs an image from the parsed ImageData
/// Usefull for debugging
impl DataVector {
    pub fn new(data: Matrix, label: u8) -> Self {
        DataVector { data, label }
    }

    pub fn expected_matrix(&self) -> Matrix {
        let dims = self.data.get_dims();
        let mut m = Matrix::zeros(dims.0, dims.1);
        m[(self.label as usize, 0)] = 1.0;
        m
    }
}
