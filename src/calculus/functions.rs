// Nomalization Functions

use std::f64::consts::{E, PI};

pub struct Function {
    pub activate: fn(&mut f64),
    pub derive: fn(&mut f64),
}

impl Function {
    pub fn sigmoid() -> Self {
        Function {
            activate: sigmoid,
            derive: sigmoid_derivative,
        }
    }

    pub fn swish() -> Self {
        Function {
            activate: swish,
            derive: swish_derivative,
        }
    }

    pub fn arctan() -> Self {
        Function {
            activate: arctan,
            derive: arctan_derivative,
        }
    }

    pub fn normal_arctan() -> Self {
        Function {
            activate: normalized_arctan,
            derive: normalized_arctan_derivative,
        }
    }

    pub fn relu() -> Self {
        Function {
            activate: relu,
            derive: relu_derivative,
        }
    }

    pub fn leaky_relu() -> Self {
        Function {
            activate: leaky_relu,
            derive: leaky_relu_derivative,
        }
    }
}
fn sigmoid(t: &mut f64) {
    *t = 1.0 / (1.0 + E.powf(-*t))
}

fn sigmoid_derivative(t: &mut f64) {
    sigmoid(t);
    *t = *t * (1.0 - *t);
}

fn swish(t: &mut f64) {
    *t = *t / (1.0 + E.powf(-*t))
}

fn swish_derivative(t: &mut f64) {
    *t = (1.0 + E.powf(-*t) + *t * E.powf(-*t)) / (1.0 + E.powf(-*t)).powf(2.0)
}

fn arctan(t: &mut f64) {
    *t = t.atan();
}

fn arctan_derivative(t: &mut f64) {
    *t = 1.0 / (t.powi(2) + 1.0);
}

fn normalized_arctan(t: &mut f64) {
    *t = t.atan() / (PI / 2.0);
}

fn normalized_arctan_derivative(t: &mut f64) {
    // normalized_arctan(t); //why???? does not work without this
    *t = (1.0 / (t.powi(2) + 1.0)) / (PI / 2.0);
}

fn relu(t: &mut f64) {
    *t = t.max(0.0)
}

fn relu_derivative(t: &mut f64) {
    if *t < 0.0 {
        *t = 0.0;
        return;
    }

    *t = 1.0;
}

fn leaky_relu(t: &mut f64) {
    *t = t.max(0.1 * *t)
}

fn leaky_relu_derivative(t: &mut f64) {
    if *t < 0.0 {
        *t = 0.1;
        return;
    }

    *t = 1.0;
}
