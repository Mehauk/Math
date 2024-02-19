// Nomalization Functions

use std::f64::consts::{E, PI};

pub struct Function {
    activator: fn(&mut f64),
    derivative: fn(&mut f64),
}

impl Function {
    pub fn activate(&self) -> fn(&mut f64) {
        self.activator
    }
    pub fn derive(&self) -> fn(&mut f64) {
        self.derivative
    }
}

impl Function {
    pub fn sigmoid() -> Self {
        Function {
            activator: sigmoid,
            derivative: sigmoid_derivative,
        }
    }

    pub fn swish() -> Self {
        Function {
            activator: swish,
            derivative: swish_derivative,
        }
    }

    pub fn arctan() -> Self {
        Function {
            activator: arctan,
            derivative: arctan_derivative,
        }
    }

    pub fn normal_arctan() -> Self {
        Function {
            activator: normalized_arctan,
            derivative: normalized_arctan_derivative,
        }
    }

    pub fn relu() -> Self {
        Function {
            activator: relu,
            derivative: relu_derivative,
        }
    }

    pub fn leaky_relu() -> Self {
        Function {
            activator: leaky_relu,
            derivative: leaky_relu_derivative,
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
