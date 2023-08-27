use image;
use nalgebra::{ArrayStorage, Const, Matrix};
use std::{
    fs::File,
    io::{Error, Read},
};

pub struct DataSet<const R: usize, const C: usize> {
    pub training_data: Vec<ImageData<R, C>>,
    pub testing_data: Vec<ImageData<R, C>>,
}

impl<const R: usize, const C: usize> DataSet<R, C> {
    pub fn load_data(
        dataset_path: &str,
        dataset_name: &str,
        parse_data: impl Fn(&str, &str, &str) -> Result<Vec<ImageData<R, C>>, Error>,
    ) -> Result<DataSet<R, C>, Error> {
        let _test_data = parse_data(dataset_path, dataset_name, "test")?;
        let _train_data = parse_data(dataset_path, dataset_name, "train")?;

        Ok(DataSet {
            testing_data: _test_data,
            training_data: _train_data,
        })
    }
}

pub struct ImageData<const R: usize, const C: usize> {
    /// values between 0-1 (inc), normalized from u8
    pub pixels: Matrix<f64, Const<R>, Const<C>, ArrayStorage<f64, R, C>>,
    pub label: u8,

    // property used for recontructing the image
    _dims: Option<(i32, i32)>,
}

/// Contructs an image from the parsed ImageData
/// Usefull for debugging
impl<const R: usize, const C: usize> ImageData<R, C> {
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

/// parses mnist ubyte files
///
/// ### Args
/// - `dataset_path`: path_to_the_file/
/// - `dataset_name`: base-name
/// - `ext`: -test-labels
///
/// ### Example
/// ```
/// parse_mnist("src/assets/machine_learning/", "letters", "train")
/// ```
pub fn parse_mnist(dataset_path: &str, dataset_name: &str, ext: &str) -> Result<(), Error> {
    let mut image_file = File::open(format!("{}{}.{}.images", dataset_path, dataset_name, ext))?;
    let mut label_file = File::open(format!("{}{}.{}.labels", dataset_path, dataset_name, ext))?;

    // MNIST dataset documentation:
    // [0, 0, 8, n]; are the first 4 bytes.
    // the first two are alwyas 0,
    // followed by a byte that represent the type of data; 8 -> u8,
    // followed by n, which represents the dimensions
    // of the stored data.
    let mut magic_number_images: [u8; 4] = [0; 4];
    let mut magic_number_labels: [u8; 4] = [0; 4];
    image_file.read_exact(&mut magic_number_images)?;
    label_file.read_exact(&mut magic_number_labels)?;

    // will hold the sizes of the dimensions
    let mut image_sizes = Vec::<i32>::new();
    let mut label_sizes = Vec::<i32>::new();

    // the next n * 4 bytes will be pushed as i32 into sizes
    let mut int_bytes: [u8; 4] = [0; 4];
    for _ in 0..magic_number_images[3] {
        image_file.read_exact(&mut int_bytes)?;
        image_sizes.push(i32::from_be_bytes(int_bytes));
    }

    for _ in 0..magic_number_labels[3] {
        label_file.read_exact(&mut int_bytes)?;
        label_sizes.push(i32::from_be_bytes(int_bytes));
    }

    print!("{:?}", image_sizes);

    let (width, height) = (image_sizes[1], image_sizes[2]);

    let image_dimensions = width * height;

    // let mut image_vector = Vec::<ImageData>::new();

    let mut buf: [u8; 1] = [0; 1];

    for _ in 0..image_sizes[0] {
        let mut pixels = Vec::<f64>::new();
        for _ in 0..image_dimensions {
            image_file.read_exact(&mut buf)?;
            pixels.push(buf[0] as f64 / 255.0);
        }

        label_file.read_exact(&mut buf)?;
        let label = buf[0];
        // image_vector.push(ImageData {
        //     pixels,
        //     // 1-26 (inc) -> for letters (a-z)
        //     // 0-9 (inc) -> for digits
        //     label,
        //     _dims: Some((width, height)),
        // });

        println!("Contructed ImageData: {}", label)
    }

    Ok(())
    // Ok(image_vector)
}
