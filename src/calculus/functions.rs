// Nomalization Functions

use std::f64::consts::{E, PI};
pub struct Function {
    pub calc: Box<dyn FnMut(&mut f64) -> ()>,
    pub derive: Box<dyn FnMut(&mut f64) -> ()>,
}

pub fn _normalized_arctan(t: &mut f64) {
    *t = t.atan() / (PI / 2.0);
}

pub fn _normalized_arctan_derivative(t: &mut f64) {
    *t = (1.0 / (t.powi(2) + 1.0)) / (PI / 2.0);
}

pub fn sigmoid(t: &mut f64) {
    *t = 1.0 / (1.0 + E.powf(-*t))
}

pub fn sigmoid_derivative(t: &mut f64) {
    // _sigmoid(t); // wont be necessary as sigmoid will already be calculated
    *t = *t * (1.0 - *t);
}
