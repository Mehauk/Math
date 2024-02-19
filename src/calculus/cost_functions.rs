pub struct CostFunction {
    cost: fn(&mut f64),
    derivative: fn(&mut f64),
}

impl CostFunction {
    pub fn calc_cost(&self) -> fn(&mut f64) {
        self.cost
    }

    pub fn derive(&self) -> fn(&mut f64) {
        self.derivative
    }
}

impl CostFunction {
    pub fn quadratic() -> Self {
        CostFunction {
            cost: quadratic_cost,
            derivative: quadratic_cost_derivative,
        }
    }
}

/// calculates the cost the nueral network; `C = (R - E)^2`
/// - `C` cost Matrix
/// - `R - E` Difference of actual result verses expected
pub fn quadratic_cost(t: &mut f64) {
    *t *= *t;
}

/// calculates the derivative of the cost; `C' = 2(R - E)`
/// - `C'` cost derivative Matrix
/// - `R - E` Difference of actual result verses expected
pub fn quadratic_cost_derivative(t: &mut f64) {
    *t = 2.0 * *t;
}
