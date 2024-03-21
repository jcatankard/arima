mod difference;
mod lags;

use super::Model;
use std::cmp::max;
use numpy::ndarray::{Array, Array1, Array2, Axis, concatenate, s};


impl Model {
    pub(super) fn integrate_predictions(&self, y_preds: &Array1<f64>, endog_fit: &Array1<f64>) -> Array1<f64> {
        let intercept = self.coefs.as_ref().unwrap()[0];
        let mut y_preds = y_preds - intercept;
        y_preds = difference::integrate_all(&y_preds, endog_fit, self.order.d, self.seasonal_order.d, self.seasonal_order.s);
        y_preds + intercept
    }
}

impl Model {
    pub(super) fn unwrap_x(&self, x: Option<&Array2<f64>>, default_length: usize) -> Array2<f64> {
        let x = x.unwrap_or(&Array::zeros((default_length, 0))).to_owned();
        self.check_x_size(default_length, &x);
        x
    }

    pub(super) fn difference_xy(
        &self,
        exog_fit: &Array2<f64>,
        exog_future: &Array2<f64>,
        endog_fit: &Array1<f64>,
        h: usize
    ) -> (Array2<f64>, Array1<f64>) {

        let exog = concatenate![Axis(0), exog_fit.view(), exog_future.view()];
        let exog_diff = difference::diff_all2d(&exog, self.order.d, self.seasonal_order.d, self.seasonal_order.s);

        let mut endog_diff = difference::diff_all1d(&endog_fit, self.order.d, self.seasonal_order.d, self.seasonal_order.s);
        endog_diff = concatenate![Axis(0), endog_diff.view(), Array::zeros(h).view()];

        (exog_diff, endog_diff)
    }

    pub(super) fn prepare_xy(&self, exog: &Array2<f64>, endog: &Array1<f64>) -> (Array2<f64>, Array1<f64>) {

        let nobs_lost = max(self.order.p, self.seasonal_order.p * self.seasonal_order.s);
        if nobs_lost >= endog.len() {
            panic!("y used for fitting is not long enough based on model specification.")
        }
        let nobs = endog.len() - nobs_lost;

        let x = self.prepare_x(&exog, &endog, nobs);
        let y = endog.slice(s![-(nobs as isize)..]).to_owned();
        (x, y)
    }

    fn prepare_x(&self, exog: &Array2<f64>, endog: &Array1<f64>, nobs: usize) -> Array2<f64> {

        let y_lags = lags::create_lags(&endog, self.order.p, self.order.s);
        let y_lags_seasonal = lags::create_lags(&endog, self.seasonal_order.p, self.seasonal_order.s);

        let errors: Array2<f64> = Array::zeros((nobs, self.order.q));
        let errors_seasonal: Array2<f64> = Array::zeros((nobs, self.seasonal_order.q));
        let intercept: Array2<f64> = Array::ones((nobs, 1));

        let nobs = nobs as isize;
        concatenate![
            Axis(1),
            intercept.view(),
            errors.view(),
            errors_seasonal.view(),
            y_lags.slice(s![-nobs.., ..]),
            y_lags_seasonal.slice(s![-nobs.., ..]),
            exog.slice(s![-nobs.., ..])
        ]
    }

    fn check_x_size(&self, size: usize, x: &Array2<f64>) {
        if x.shape()[0] != size {
            panic!("x is length: {}. It should be length: {}.", x.shape()[0], size);
        }

        if let Some(x_fit) = &self.exog_fit {
            if x.shape()[1] != x_fit.shape()[1] {
                panic!("x has {} columns. It should have {}.", x.shape()[1], x_fit.shape()[1]);
            }
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{Array, Array1, Array2, arr1, arr2};
    // run with "cargo test -- --show-output" to see output

    #[test]
    #[should_panic(expected = "y used for fitting is not long enough based on model specification")]
    fn prepare_data_y_too_small() {
        let model = Model::sarima((2, 1, 3), (1, 1, 1, 7));
        let y = arr1(&[0., 1., 2., 3.]);
        let x: Array2<f64> = Array::zeros((y.len(), 0));
        model.prepare_xy(&x, &y);
    }

    #[test]
    #[should_panic(expected = "columns. It should have")]
    fn prepare_data_x_future_wrong_cols() {
        
        let h = 10;
        let y: Array1<f64> = Array::ones(200);
        let x: Array2<f64> = Array::ones((200, 10));
        let x_future: Array2<f64> = Array::ones((h, 8));

        let mut model = Model::moving_average(0);
        model.forecast(&y, h, Some(&x), Some(&x_future));
    }

    #[test]
    #[should_panic(expected = "It should be length")]
    fn prepare_data_y_len_not_equal_x_len() {
        let model = Model::sarima((1, 1, 0), (2, 2, 0, 2));
        let y = arr1(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let x: Array2<f64> = arr2(&[[0., 1., 2., 3., 4.], [0., 1., 2., 3., 4.]]).t().to_owned();
        model.unwrap_x(Some(&x), y.len());
    }
}