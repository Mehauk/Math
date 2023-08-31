// Nomalization Functions

pub fn _normalized_arctan(t: &mut f64) {
    *t = t.atan() / (std::f64::consts::PI / 2.0);
}
