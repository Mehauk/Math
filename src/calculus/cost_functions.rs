pub struct CostFunction {
    pub cost: fn(&mut f64),
    pub derive: fn(&mut f64),
}

// /// calculates the cost the nueral network; `C = (R - E)^2`
// /// - `C` cost Matrix
// /// - `R` resulting output Matrix of network
// /// - `E` expected output Matrix contructed from label
// pub fn _cost(result_matrix: &Matrix, label: u8) -> Matrix {
//     let mut m = result_matrix.clone();
//     m[(label as usize, 0)] -= 1.0;
//     m.apply_into(|x| *x *= *x)
// }

// /// calculates the derivative of the cost; `C' = 2(R - E)
// /// - `C` cost Matrix
// /// - `R` resulting output Matrix of network
// /// - `E` expected output Matrix contructed from label
// pub fn cost_derivative(result_matrix: &Matrix, label: u8) -> Matrix {
//     let mut m = result_matrix.clone();
//     m[(label as usize, 0)] -= 1.0;
//     m = m * 2.0;
//     m
// }
