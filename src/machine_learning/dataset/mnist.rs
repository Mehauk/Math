use std::{
    fs::File,
    io::{Error, Read},
};

use super::ImageData;

pub const INPUT_SIZE: usize = 28 * 28;

/// parses mnist ubyte files
///
/// ### Args
/// - `dataset_path`: path_to_the_file/
/// - `dataset_name`: base-name
/// - `ext`: -test-labels
///
/// ### Example
/// ```
/// parse_mnist::<{INPUT_SIZE}, 1>("src/assets/machine_learning/", "letters", "train")
/// ```
pub fn parse_mnist<const I: usize>(
    dataset_path: &str,
    dataset_name: &str,
    ext: &str,
) -> Result<Vec<ImageData<I>>, Error> {
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

    // these are expeected to be 28, 28
    let (width, height) = (image_sizes[1], image_sizes[2]);

    let image_dimensions = (width * height) as usize;
    let mut image_vector = Vec::<ImageData<I>>::new();

    let mut buf: [u8; 1] = [0; 1];

    println!("Contructing {}.{} Dataset: ", dataset_name, ext);
    for i in 0..image_sizes[0] {
        let mut pixels = Vec::<f64>::new();
        for _ in 0..image_dimensions {
            image_file.read_exact(&mut buf)?;
            pixels.push(buf[0] as f64 / 255.0);
        }

        label_file.read_exact(&mut buf)?;
        let label = buf[0];
        image_vector.push(ImageData {
            pixels: nalgebra::DMatrix::<f64>::from_vec(I, 1, pixels),

            // 1-26 (inc) -> for letters (a-z)
            // 0-9 (inc) -> for digits
            label,
            _dims: Some((width, height)),
        });

        let fraction = ((i + 1) as f32) / (image_sizes[0] as f32) * 100.0;
        let mut loading_indicator: [char; 10] = ['_'; 10];
        for li in 0..(fraction as usize / 10) {
            loading_indicator[li] = '█'
        }
        print!(
            "\rcurrent({:?}) - {}  {}/{}  {:.2}%     ",
            _letter_from_number(&label),
            loading_indicator.iter().collect::<String>(),
            i + 1,
            image_sizes[0],
            fraction,
        );
    }
    println!("\nDone!");

    Ok(image_vector)
}

fn _letter_from_number(n: &u8) -> Option<char> {
    let (a, b) = (b'a', b'b');
    let c = a + (b - a) * (n - 1);
    if c <= b'z' {
        return Some(c as char);
    }

    None
}
