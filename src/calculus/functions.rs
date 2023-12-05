// Nomalization Functions

use std::f64::consts::{E, PI};

pub struct Function {
    pub calc: fn(&mut f64),
    pub derive: fn(&mut f64),
}

pub fn _normalized_arctan(t: &mut f64) {
    let x = _normalized_arctan;
    *t = t.atan() / (PI / 2.0);
}

pub fn _normalized_arctan_derivative(t: &mut f64) {
    *t = (1.0 / (t.powi(2) + 1.0)) / (PI / 2.0);
}

pub fn _sigmoid(t: &mut f64) {
    *t = 1.0 / (1.0 + E.powf(-*t))
}

pub fn _sigmoid_derivative(t: &mut f64) {
    // _sigmoid(t); // wont be necessary as sigmoid will already be calculated
    *t = *t * (1.0 - *t);
}
