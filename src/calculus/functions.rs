// Nomalization Functions

use std::f64::consts::E;

pub fn _normalized_arctan(t: &mut f64) {
    *t = t.atan() / (std::f64::consts::PI / 2.0);
}

pub fn _sigmoid(t: &mut f64) {
    *t = 1.0 / (1.0 + E.powf(-*t))
}
