// Nomalization Functions

use std::f64::consts::{E, PI};

pub struct Function {
    pub calc: fn(&mut f64),
    pub derive: fn(&mut f64),
}

impl Function {
    pub fn sigmoid() -> Self {
        Function {
            calc: sigmoid,
            derive: sigmoid_derivative,
        }
    }

    pub fn normal_arctan() -> Self {
        Function {
            calc: normalized_arctan,
            derive: normalized_arctan_derivative,
        }
    }
}

fn normalized_arctan(t: &mut f64) {
    *t = t.atan() / (PI / 2.0);
}

fn normalized_arctan_derivative(t: &mut f64) {
    *t = (1.0 / (t.powi(2) + 1.0)) / (PI / 2.0);
}

fn sigmoid(t: &mut f64) {
    *t = 1.0 / (1.0 + E.powf(-*t))
}

fn sigmoid_derivative(t: &mut f64) {
    // _sigmoid(t); // wont be necessary as sigmoid will already be calculated
    *t = *t * (1.0 - *t);
}
