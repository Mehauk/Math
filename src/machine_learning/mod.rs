use std::{
    fs::File,
    io::{Error, ErrorKind, Read},
};

pub fn load_data(dataset_path: &str, dataset_name: &str) -> Result<DataSet, Error> {
    let _test_images = read_data_from_file(dataset_path, dataset_name, "-test-images")?;
    let _test_labels = read_data_from_file(dataset_path, dataset_name, "-test-labels")?;
    let _train_images = read_data_from_file(dataset_path, dataset_name, "-train-images")?;
    let _train_labels = read_data_from_file(dataset_path, dataset_name, "-train-labels")?;

    Ok(DataSet {
        training_data: vec![],
        testing_data: vec![],
    })
}

fn read_data_from_file(
    dataset_path: &str,
    dataset_name: &str,
    ext: &str,
) -> Result<Vec<ImageData>, Error> {
    let mut file = File::open(format!("{}{}{}", dataset_path, dataset_name, ext))?;

    let mut magic_number: [u8; 4] = [0; 4];
    file.read_exact(&mut magic_number)?;

    let mut sizes = Vec::<i32>::new();
    
    for _ in 1..magic_number[3] {
        
    }

    println!("{:#?}", magic_number);
    Err(Error::from(ErrorKind::AddrInUse))
}

pub struct DataSet {
    pub training_data: Vec<ImageData>,
    pub testing_data: Vec<ImageData>,
}

pub struct ImageData {
    pub image: Vec<f64>,
    pub label: u8,
}
