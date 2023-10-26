use image;
use nalgebra::SMatrix;
use std::io::Error;

pub mod mnist;

pub struct DataSet<const I: usize> {
    pub training_data: Vec<ImageData<I>>,
    pub testing_data: Vec<ImageData<I>>,
}

impl<const I: usize> DataSet<I> {
    pub fn load_data(
        dataset_path: &str,
        dataset_name: &str,
        parse_data: impl Fn(&str, &str, &str) -> Result<Vec<ImageData<I>>, Error>,
    ) -> Result<DataSet<I>, Error> {
        let _test_data = parse_data(dataset_path, dataset_name, "test")?;
        let _train_data = parse_data(dataset_path, dataset_name, "train")?;

        Ok(DataSet {
            testing_data: _test_data,
            training_data: _train_data,
        })
    }
}

// TODO: convert to Trait?
pub struct ImageData<const I: usize> {
    /// `ArrayStorage` requires constant usize for Row and Column.
    ///
    /// We could also use `VecStorage` to create a matrix with
    /// dimensions calculated at runtime, but operations would be slower.
    ///
    /// values in the matrix are between 0-1 (inc), normalized from u8
    pub pixels: SMatrix<f64, I, 1>,
    pub label: u8,

    // property used for recontructing the image
    _dims: Option<(i32, i32)>,
}

/// Contructs an image from the parsed ImageData
/// Usefull for debugging
impl<const I: usize> ImageData<I> {
    pub fn _new(pixels: SMatrix<f64, I, 1>, label: u8) -> Self {
        ImageData {
            pixels,
            label,
            _dims: None,
        }
    }

    fn _show(&self, file_path: &str) {
        if let Some((width, height)) = self._dims {
            let width = width as u32;
            let height = height as u32;
            let mut image = image::RgbImage::new(width, height);
            for (i, e) in self.pixels.iter().enumerate() {
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
