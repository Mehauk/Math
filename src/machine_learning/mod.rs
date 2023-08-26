use std::{
    fmt::Debug,
    fs::File,
    io::{Error, ErrorKind, Read},
};

pub struct DataSet {
    pub training_data: Vec<ImageData>,
    pub testing_data: Vec<ImageData>,
}

impl DataSet {
    pub fn load_data(
        dataset_path: &str,
        dataset_name: &str,
        parse_data: impl Fn(&str, &str, &str) -> Result<Vec<ImageData>, Error>,
    ) -> Result<DataSet, Error> {
        let _test_data = parse_data(dataset_path, dataset_name, "-test")?;
        let _train_data = parse_data(dataset_path, dataset_name, "-train")?;

        Ok(DataSet {
            training_data: vec![],
            testing_data: vec![],
        })
    }
}

pub struct ImageData {
    /// a value between 0-1, normalized from u8
    pub pixels: Vec<f64>,
    pub label: u8,
}

impl Debug for ImageData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s: String = String::new();
        s.push('\n');
        s.push_str("This is supposed to be a: ");
        s.push_str(&self.label.to_string());
        s.push('\n');
        for i in 0..28 {
            for x in 0..28 {
                if self.pixels[i * x] < 0.51 {
                    s.push('.')
                } else {
                    s.push('8')
                }
            }
            s.push('\n');
        }
        f.write_str(&s)
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
/// parse_mnist("src/assets/machine_learning/", "letters", "-train")
/// ```
pub fn parse_mnist(
    dataset_path: &str,
    dataset_name: &str,
    ext: &str,
) -> Result<Vec<ImageData>, Error> {
    let mut image_file = File::open(format!("{}{}{}-images", dataset_path, dataset_name, ext))?;
    let mut label_file = File::open(format!("{}{}{}-labels", dataset_path, dataset_name, ext))?;

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

    let image_dimensions = image_sizes[1] * image_sizes[2];

    let mut image_vector = Vec::<ImageData>::new();

    let mut buf: [u8; 1] = [0; 1];

    for _ in 0..image_sizes[0] {
        let mut pixels = Vec::<f64>::new();
        for _ in 0..image_dimensions {
            image_file.read_exact(&mut buf)?;
            pixels.push(buf[0] as f64 / 255.0);
        }

        label_file.read_exact(&mut buf)?;
        let label = buf[0];
        image_vector.push(ImageData { pixels, label });

        println!("{:?}", image_vector[0]);

        return Err(Error::from(ErrorKind::AddrInUse));
    }

    Ok(image_vector)
}
