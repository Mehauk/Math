// Nomalization Functions

use std::f64::consts::{E, PI};

pub trait Function {
    fn calc(&self, x: &mut f64);
    fn derive(&self, x: &mut f64);
}

pub struct Sigmoid {}

impl Function for Sigmoid {
    fn calc(&self, x: &mut f64) {
        _sigmoid(x)
    }

    fn derive(&self, x: &mut f64) {
        _sigmoid_derivative(x)
    }
}

pub struct NormalizedArctan {}

impl Function for NormalizedArctan {
    fn calc(&self, x: &mut f64) {
        _normalized_arctan(x)
    }

    fn derive(&self, x: &mut f64) {
        _normalized_arctan_derivative(x)
    }
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
