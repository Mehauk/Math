use image;
use nalgebra::DVector;
use std::io::Error;

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

// TODO: convert to Trait?
pub struct DataVector {
    /// `ArrayStorage` requires constant usize for Row and Column.
    ///
    /// We could also use `VecStorage` to create a matrix with
    /// dimensions calculated at runtime, but operations would be slower.
    ///
    /// values in the matrix are between 0-1 (inc), normalized from u8
    pub data: DVector<f64>,
    pub label: u8,

    // property used for recontructing the image
    _dims: Option<(i32, i32)>,
}

/// Contructs an image from the parsed ImageData
/// Usefull for debugging
impl DataVector {
    pub fn _new(data: DVector<f64>, label: u8) -> Self {
        DataVector {
            data,
            label,
            _dims: None,
        }
    }

    pub fn _show(&self, file_path: &str) {
        if let Some((width, height)) = self._dims {
            let width = width as u32;
            let height = height as u32;
            let mut image = image::RgbImage::new(width, height);
            for (i, e) in self.data.iter().enumerate() {
                let p = (e * 255.0) as u8;
                image.put_pixel(i as u32 / width, i as u32 % height, image::Rgb([p, p, p]));
            }
            if let Ok(_) = image.save(file_path) {
                println!("Saved imageData to: {}", file_path);
                return;
            };
        }

        println!("Failed to save imageData");
    }
}
