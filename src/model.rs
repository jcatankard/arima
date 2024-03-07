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
}