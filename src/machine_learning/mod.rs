pub fn load_data(dataset_path: &str, dataset_name: &str) -> Result<DataSet, std::io::Error> {
    let _test_images = std::fs::read(format!("{}{}-test-images", dataset_path, dataset_name))?;
    let _test_labels = std::fs::read(format!("{}{}-test-labels", dataset_path, dataset_name))?;
    let _train_images = std::fs::read(format!("{}{}-train-images", dataset_path, dataset_name))?;
    let _train_labels = std::fs::read(format!("{}{}-train-labels", dataset_path, dataset_name))?;

    // println!("{:?}", _test_images);

    Ok(DataSet {
        training_data: vec![],
        testing_data: vec![],
    })
}

pub struct DataSet {
    pub training_data: Vec<ImageData>,
    pub testing_data: Vec<ImageData>,
}

pub struct ImageData {
    pub image: Vec<f64>,
    pub label: u8,
}
