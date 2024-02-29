use crate::linear_algebra::Matrix;

pub struct CostFunction {
    cost: fn(&Matrix, &Matrix) -> Matrix,
    derivative: fn(&Matrix, &Matrix) -> Matrix,
}

impl CostFunction {
    pub fn calc_cost(&self) -> fn(&Matrix, &Matrix) {
        self.cost
    }

    pub fn derive(&self) -> fn(&Matrix, &Matrix) {
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
fn quadratic_cost(t: &mut f64) {
    *t *= *t;
}

/// calculates the derivative of the cost; `C' = 2(R - E)`
/// - `C'` cost derivative Matrix
/// - `R - E` Difference of actual result verses expected
fn quadratic_cost_derivative(t: &mut f64) {
    *t = 2.0 * *t;
}

// /// calculates the cost the nueral network; `C = E * ln(R) + (1 - E) ln(1 - R)`
// /// - `C` cost Matrix
// /// - `R - E` Difference of actual result verses expected
// fn cross_entropy_cost(t: &mut f64) {
//     *t *= *t;
// }

// /// calculates the derivative of the cost; `C' = ln(R) + *E/R -ln(1 - R) + (1 - E)/(1 - R)`
// /// - `C'` cost derivative Matrix
// /// - `R - E` Difference of actual result verses expected
// fn cross_entropy_cost_derivative(t: &mut f64) {
//     *t = 2.0 * *t;
// }
