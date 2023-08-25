use std::{
    fs::File,
    io::{Error, ErrorKind, Read},
};

pub struct ImageData {
    pub image: Vec<f64>,
    pub label: Vec<u8>,
}

pub struct DataSet {
    pub training_data: Vec<ImageData>,
    pub testing_data: Vec<ImageData>,
}

impl DataSet {
    pub fn load_data(
        dataset_path: &str,
        dataset_name: &str,
        data_parser: &impl ParseDataset,
    ) -> Result<DataSet, Error> {
        let _test_images = data_parser.from_file(dataset_path, dataset_name, "-test-images")?;
        let _test_labels = data_parser.from_file(dataset_path, dataset_name, "-test-labels")?;
        let _train_images = data_parser.from_file(dataset_path, dataset_name, "-train-images")?;
        let _train_labels = data_parser.from_file(dataset_path, dataset_name, "-train-labels")?;

        Ok(DataSet {
            training_data: vec![],
            testing_data: vec![],
        })
    }
}

pub struct MnistDatasetParser {}

impl ParseDataset for MnistDatasetParser {
    fn from_file(
        &self,
        dataset_path: &str,
        dataset_name: &str,
        ext: &str,
    ) -> Result<Vec<ImageData>, Error> {
        let mut file = File::open(format!("{}{}{}", dataset_path, dataset_name, ext))?;

        // According to the documention of the MNIST dataset
        let mut magic_number: [u8; 4] = [0; 4];
        file.read_exact(&mut magic_number)?;

        let mut sizes = Vec::<i32>::new();

        for _ in 1..magic_number[3] {
            let mut int_bytes: [u8; 4] = [0; 4];
            file.read_exact(&mut int_bytes)?;
            sizes.push(i32::from_be_bytes(int_bytes));
        }

        println!("{:#?}", magic_number);
        Err(Error::from(ErrorKind::AddrInUse))
    }
}

pub trait ParseDataset {
    fn from_file(
        &self,
        dataset_path: &str,
        dataset_name: &str,
        ext: &str,
    ) -> Result<Vec<ImageData>, Error>;
}
