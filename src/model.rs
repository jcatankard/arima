mod prepare_data;
mod fit_predict;
mod new;
use numpy::ndarray::{Array1, Array2};
use pyo3::{pyclass, pymethods};

// https://github.com/PyO3/pyo3/blob/main/guide/pyclass_parameters.md
#[derive(Debug)]
#[pyclass(name = "Model", module = "arima")]
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
// #[pymethods]
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
    /// 

    // #[new]
    pub fn sarima(order: (usize, usize, usize), seasonal_order: (usize, usize, usize, usize)) -> Self {
        let (p, d, q) = order;
        let order = Order {p, d, q, s: 1};

        let (p, d, q, s) = seasonal_order;
        if s == 1 {panic!("It doesn't make sense for periodicity (s) to be set to 1.")}
        let seasonal_order = Order {p, d, q, s};

        let error_model = if order.q + seasonal_order.q == 0 {None} else {
            let order = Order {d: 0, q: 0, ..order};
            let seasonal_order = Order {d: 0, q: 0, ..seasonal_order};
            let model = Self {order, seasonal_order, y_original: None, y_fit: None, x_fit: None, coefs_fit: None, errors_fit: None, error_model: None};
            Some(Box::new(model))
        };

        Self {order, seasonal_order, y_original: None, y_fit: None, x_fit: None, coefs_fit: None, errors_fit: None, error_model}
    }

    /// Create an [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) model
    /// - p: AR(p) auto regressive terms
    /// - d: I(d) integrated terms
    /// - q: MA(q) moving average terms
    /// 

    pub fn arima(p: usize, d: usize, q: usize) -> Self {
        Self::sarima((p, d, q), (0, 0, 0, 0))
    }

    /// Create an [ARMA](https://en.wikipedia.org/wiki/Autoregressive_moving-average_model) model
    /// - p: AR(p) auto regressive terms
    /// - q: MA(q) moving average terms
    ///

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

        let coefs = model.coefs_fit.as_ref().unwrap().mapv(|x| (100. * x).round() / 100.);

        assert_eq!(arr1(&[cons, lag1, lag2]), coefs);

        let y_preds = model.predict(20, None).mapv(|x| (100. * x).round() / 100.);
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

        let coefs = model.coefs_fit.as_ref().unwrap().mapv(|x| (100. * x).round() / 100.);

        assert_eq!(arr1(&[cons, lag1, lag2, lag_s]), coefs);

        let y_preds = model.predict(20, None).mapv(|x| (100. * x).round() / 100.);
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

        let coefs = model.coefs_fit.as_ref().unwrap().mapv(|x| (100. * x).round() / 100.);

        assert_eq!(x_coefs, coefs.slice(s![1..]));

        let y_preds = model.predict(20, Some(&x_test)).mapv(|x| (100. * x).round() / 100.);
        y_test = y_test.mapv(|x| (100. * x).round() / 100.);
        assert_eq!(y_test, y_preds);
    }

    #[test]
    #[should_panic(expected = "to be set to 1")]
    fn new_seasonal_s_equal_one() {
        let _model = Model::sarima((1, 2, 3), (4, 5, 6, 1)); 
    }

    #[test]
    fn new_sarima() {
        let model = Model::sarima((1, 2, 3), (4, 5, 6, 7));
        assert_eq!(model.order, Order {p: 1, d: 2, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 4, d: 5, q: 6, s: 7});
    }

    #[test]
    fn new_arima() {
        let model = Model::arima(1, 2, 3);
        assert_eq!(model.order, Order {p: 1, d: 2, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn new_arma() {
        let model = Model::arma(1, 3);
        assert_eq!(model.order, Order {p: 1, d: 0, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn new_ar() {
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