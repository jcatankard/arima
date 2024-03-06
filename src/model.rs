mod prepare_data;
mod fit_predict;
mod new;
use numpy::ndarray::{Array1, Array2};

#[derive(Debug)]
pub struct Model {
    // order: (AR(p), I(d), MA(q), 1)
    // seasonal_order: (AR(p), I(d), MA(q), s)
    // x_fit: data used for fitting incl. exongenous variables, lags and error terms
    // coefs_fit: last coefficients from fitting
    // errors_fit: y - ŷ
    // error_model: forecasting future errors for MA models
    order: Order,
    seasonal_order: Order,
    y_original: Option<Array1<f64>>,
    y_fit: Option<Array1<f64>>,
    x_fit: Option<Array2<f64>>,
    coefs_fit: Option<Array1<f64>>,
    errors_fit: Option<Array1<f64>>,
    error_model: Option<Box<Model>>
}

/// p: AR (auto regressive) terms
/// d: I (integrated) terms
/// q: MA (moving average) terms
/// s: periodicity
#[derive(PartialEq, Debug)]
struct Order {
    p: usize,
    d: usize,
    q: usize,
    s: usize
}

/// # Train and forecast
/// 
impl Model {
    /// - y: timeseries
    /// - x: exogenous variables, same length as y
    pub fn fit(&mut self, y: &Array1<f64>, x: Option<&Array2<f64>>) {
        self.y_original = Some(y.to_owned());

        let (y, mut x) = self.prepare_for_fit(&y, x);
        let (coefs, errors) = self.fit_internal(&y, &mut x);

        self.fit_error_model(&errors, &x);
        self.y_fit = Some(y);
        self.x_fit = Some(x);
        self.coefs_fit = Some(coefs);
        self.errors_fit = Some(errors);
    }

    /// - h: horizons to forecast
    /// - x: future exongenous variables, same length as h
    /// 
    /// returns predictions for h horizons
    pub fn predict(&self, h: usize, x: Option<&Array2<f64>>) -> Array1<f64> {
        let future_errors = self.predict_future_errors(h, x);
        let (mut y_preds, x, coefs) = self.prepare_for_predict(h, x, future_errors);
        self.predict_internal(h, &mut y_preds, x, coefs);
        self.un_difference(&mut y_preds);
        y_preds
    }

    /// - y: timeseries
    /// - h: horizons to forecast
    /// - x: exogenous variables, same length as y
    /// - x_future: future exongenous variables, same length as h
    /// 
    /// returns predictions for h horizons
    pub fn forecast(&mut self, y: &Array1<f64>, h: usize, x: Option<&Array2<f64>>, x_future: Option<&Array2<f64>>) -> Array1<f64> {
        self.fit(&y, x);
        self.predict(h, x_future)
    }

    /// - y: timeseries
    /// - h: horizons to forecast
    /// - x: exogenous variables, same length as y
    /// - x_future: future exongenous variables, same length as h
    /// 
    /// returns predictions for h horizons
    pub fn fit_predict(&mut self, y: &Array1<f64>, h: usize, x: Option<&Array2<f64>>, x_future: Option<&Array2<f64>>) -> Array1<f64> {
        self.forecast(&y, h, x, x_future)
    }
}


#[cfg(test)]
mod tests {
    // run with "cargo test -- --show-output" to see output
    // use super::*;

    #[test]
    fn test_something() {
        assert!(true)
    }

}