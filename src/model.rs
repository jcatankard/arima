mod prepare_data;
mod fit_predict;

use numpy::ndarray::{Array1, Array2};
use pyo3::pyclass;


#[derive(Debug)]
#[pyclass(name = "Model", module = "arima")]
pub struct Model {
    // order: (AR(p), I(d), MA(q), 1)
    // seasonal_order: (AR(p), I(d), MA(q), s)
    // exog_fit: exongenous variables used for fitting
    // endog_fit: time-series
    // coefs_fit: last coefficients from fitting
    order: Order,
    seasonal_order: Order,
    endog_fit: Option<Array1<f64>>,
    exog_fit: Option<Array2<f64>>,
    pub coefs: Option<Array1<f64>>
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
        self.endog_fit = Some(y.to_owned());
        self.exog_fit = Some(self.unwrap_x(x, y.len()));
    }

    /// - h: horizons to forecast
    /// - x: future exongenous variables, same length as h
    /// 
    /// returns predictions for h horizons
    pub fn predict(&mut self, h: usize, x: Option<&Array2<f64>>) -> Array1<f64> {

        let exog_fit = self.exog_fit.as_ref().expect("Model must be fit before predict");
        let exog_future = self.unwrap_x(x, h);
        let endog_fit = self.endog_fit.as_ref().expect("Model must be fit before predict");

        let (exog_diff, endog_diff) = self.difference_xy(exog_fit, &exog_future, endog_fit, h);
        let (mut x, mut y) = self.prepare_xy(&exog_diff, &endog_diff);
        
        let (y_preds, coefs) = self.fit_predict_internal(h, &mut y, &mut x, &exog_diff);
        self.coefs = Some(coefs);
        self.integrate_predictions(&y_preds, &endog_fit)
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

    /// Create a [SARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average#Variations_and_extensions) model.
    /// - order: (p, d, q)
    ///     - p: AR(p) auto regressive terms
    ///     - d: I(d) integrated terms
    ///     - q: MA(q) moving average terms
    /// - seasonal_order: (P, D, Q, s)
    ///     - P: AR(P) auto regressive terms
    ///     - D: I(D) integrated terms
    ///     - Q: MA(Q) moving average terms
    ///     - s: periodicity
    pub fn sarima(order: (usize, usize, usize), seasonal_order: (usize, usize, usize, usize)) -> Self {
        let (p, d, q) = order;
        let order = Order {p, d, q, s: 1};

        let (p, d, q, s) = seasonal_order;
        if s == 1 {panic!("It doesn't make sense for periodicity (s) to be set to 1.")}
        let seasonal_order = Order {p, d, q, s};

        Self {order, seasonal_order, endog_fit: None, exog_fit: None, coefs: None}
    }

    /// Create an [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) model
    /// - p: AR(p) auto regressive terms
    /// - d: I(d) integrated terms
    /// - q: MA(q) moving average terms
    pub fn arima(p: usize, d: usize, q: usize) -> Self {
        Self::sarima((p, d, q), (0, 0, 0, 0))
    }

    /// Create an [ARMA](https://en.wikipedia.org/wiki/Autoregressive_moving-average_model) model
    /// - p: AR(p) auto regressive terms
    /// - q: MA(q) moving average terms
    pub fn arma(p: usize, q: usize) -> Self {
        Self::sarima((p, 0, q), (0, 0, 0, 0))
    }

    /// Create an [Autoregressive](https://en.wikipedia.org/wiki/Autoregressive_model) model
    /// - p: AR(p) auto regressive terms
    pub fn autoregressive(p: usize) -> Self {
        Self::sarima((p, 0, 0), (0, 0, 0, 0))
    }

    /// Create a [Moving averages](https://en.wikipedia.org/wiki/Moving-average_model) model
    /// - q: MA(q) moving average terms
    pub fn moving_average(q: usize) -> Self {
        Self::sarima((0, 0, q), (0, 0, 0, 0))
    }
}


#[cfg(test)]
mod tests {
    // run with "cargo test -- --show-output" to see output
    use numpy::ndarray::{Array, Array1, arr1, s};
    use super::*;

    #[test]
    fn model_autoregressive() {

        let (cons, lag1, lag2) = (100., 0.5, -0.25);

        let mut y: Array1<f64> = Array::zeros(200) + cons;
        y[0] = 150.;
        y[1] = 50.;

        for i in 2..y.len() {
            y[i] += y[i - 1] * lag1 + y[i - 2] * lag2;
        }

        let y_train = y.slice(s![..180]).to_owned();
        
        let mut y_test = y.slice(s![180..]).to_owned();
       
        let mut model = Model::autoregressive(2);
        model.fit(&y_train, None);
        let mut y_preds = model.predict(20, None);

        let coefs = model.coefs.as_ref().unwrap().mapv(|x| (100. * x).round() / 100.);
        assert_eq!(arr1(&[cons, lag1, lag2]), coefs);

        y_preds = y_preds.mapv(|x| (100. * x).round() / 100.);
        y_test = y_test.mapv(|x| (100. * x).round() / 100.);
        assert_eq!(y_test, y_preds);
    }

    #[test]
    fn model_seasonal_ar() {

        let (cons, lag1, lag2, lag_s, s) = (60., 0.45, -0.35, 0.25, 7);

        let mut y: Array1<f64> = Array::zeros(100) + cons;

        for i in s..y.len() {
            y[i] += y[i - 1] * lag1 + y[i - 2] * lag2 + y[i - s] * lag_s;
        }

        let y_train = y.slice(s![..80]).to_owned();
        let mut y_test = y.slice(s![80..]).to_owned();
       
        let mut model = Model::sarima((2, 0, 0), (1, 0, 0, s));
        model.fit(&y_train, None);
        let mut y_preds = model.predict(20, None);

        let coefs = model.coefs.as_ref().unwrap().mapv(|x| (100. * x).round() / 100.);

        assert_eq!(arr1(&[cons, lag1, lag2, lag_s]), coefs);

        y_preds = y_preds.mapv(|x| (100. * x).round() / 100.);
        y_test = y_test.mapv(|x| (100. * x).round() / 100.);
        assert_eq!(y_test, y_preds);
    }

    #[test]
    fn model_exog() {
        let n_rows = 100;

        let mut x: Array2<f64> = Array::zeros((n_rows, 4));
        x.slice_mut(s![.., 0]).assign(&Array::linspace(-100., -20., n_rows));
        x.slice_mut(s![.., 1]).assign(&Array::logspace(3., 1., 2., n_rows));
        x.slice_mut(s![.., 2]).assign(&Array::logspace(2., -1., 2., n_rows));
        x.slice_mut(s![.., 3]).assign(&Array::geomspace(100., 200., n_rows).unwrap());

        let x_coefs = arr1(&[10.4, 20.6, -10.8, 1.2]);
        let y = x.dot(&x_coefs);

        let y_train = y.slice(s![..80]).to_owned();
        let mut y_test = y.slice(s![80..]).to_owned();
       
        let x_train = x.slice(s![..80, ..]).to_owned();
        let x_test = x.slice(s![80.., ..]).to_owned();
       
        let mut model = Model::moving_average(0);
        model.fit(&y_train, Some(&x_train));
        let mut y_preds = model.predict(20, Some(&x_test));

        let coefs = model.coefs.as_ref().unwrap().mapv(|x| (100. * x).round() / 100.);

        assert_eq!(x_coefs, coefs.slice(s![1..]));

        y_preds = y_preds.mapv(|x| (100. * x).round() / 100.);
        y_test = y_test.mapv(|x| (100. * x).round() / 100.);
        assert_eq!(y_test, y_preds);
    }

    #[test]
    #[should_panic(expected = "to be set to 1")]
    fn model_new_seasonal_s_equal_one() {
        let _model = Model::sarima((1, 2, 3), (4, 5, 6, 1)); 
    }

    #[test]
    fn model_new_sarima() {
        let model = Model::sarima((1, 2, 3), (4, 5, 6, 7));
        assert_eq!(model.order, Order {p: 1, d: 2, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 4, d: 5, q: 6, s: 7});
    }

    #[test]
    fn model_new_arima() {
        let model = Model::arima(1, 2, 3);
        assert_eq!(model.order, Order {p: 1, d: 2, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn model_new_arma() {
        let model = Model::arma(1, 3);
        assert_eq!(model.order, Order {p: 1, d: 0, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn model_new_ar() {
        let model = Model::autoregressive(1);
        assert_eq!(model.order, Order {p: 1, d: 0, q: 0, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn new_ma() {
        let model = Model::moving_average(3);
        assert_eq!(model.order, Order {p: 0, d: 0, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }
}