mod difference;
mod lags;

use super::Model;
use std::cmp::max;
use numpy::ndarray::{Array, Array1, Array2, Axis, concatenate, s};


impl Model {
    pub(super) fn prepare_for_fit(&self, y: &Array1<f64>, x: Option<&Array2<f64>>) -> (Array1<f64>, Array2<f64>) {
        let mut x = self.unwrap_x(x, y.len());
        let nobs = self.find_nobs(&y, &x);
        
        let mut y = difference::difference_all(&y, self.order.d, self.seasonal_order.d, self.seasonal_order.s);

        let errors: Array2<f64> = Array::zeros((nobs as usize, self.order.q));
        let errors_seasonal: Array2<f64> = Array::zeros((nobs as usize, self.seasonal_order.q));
        
        let y_lags = lags::create_lags(&y, self.order.p, self.order.s);
        let y_lags_seasonal = lags::create_lags(&y, self.seasonal_order.p, self.seasonal_order.s);
        
        x = self.prepare_x(x, errors, errors_seasonal, y_lags, y_lags_seasonal, nobs);
        y.slice_collapse(s![-nobs..]);
        (y, x)
    }
    
    fn find_nobs(&self, y: &Array1<f64>, x: &Array2<f64>) -> isize {
        let n_lost = self.nobs_lost_from_diffs_and_lags();
        let min_rows = n_lost + self.n_model_features(&x);
        if y.len() < min_rows {
            panic!("y is of length {}. It must be at least {} based on provided model parameters.", y.len(), min_rows);
        }
        (y.len() - n_lost) as isize
    }

    fn nobs_lost_from_diffs_and_lags(&self) -> usize {
        let from_lags = max(self.order.p, self.seasonal_order.p * self.seasonal_order.s);
        let from_diffs = self.order.d + self.seasonal_order.d * self.seasonal_order.s;
        from_lags + from_diffs
    }

    fn n_model_features(&self, x: &Array2<f64>) -> usize {
        // need at least as many rows as columns for linear regression solver to work (assuming no regularization) incl intercept
        x.shape()[1]
        + self.order.q
        + self.seasonal_order.q
        + self.order.p
        + self.seasonal_order.p
        + 1  // intercept
    }
}

impl Model {
    pub(super) fn prepare_for_predict(&self, h: usize, x: Option<&Array2<f64>>, future_errors: Array1<f64>) -> (Array1<f64>, Array2<f64>, &Array1<f64>) {
        let (x_fit, errors_fit, coefs_fit, y_fit) = self.get_fit_refs();

        let errors = concatenate![Axis(0), errors_fit.view(), future_errors.view()];

        let x_future = self.prepare_x_future(h, x, errors, &coefs_fit);
        let x = concatenate![Axis(0), x_fit.view(), x_future.view()];

        let y_preds = concatenate![Axis(0), y_fit.view(), Array::zeros(h).view()];

        (y_preds, x, coefs_fit)
    }

    fn get_fit_refs(&self) -> (&Array2<f64>, &Array1<f64>, &Array1<f64>, &Array1<f64>) {
        let message: &str = "Model must be fit before predict";
        let x_fit = self.x_fit.as_ref().expect(message);
        let errors_fit = self.errors_fit.as_ref().expect(message);
        let coefs_fit = self.coefs_fit.as_ref().expect(message);
        let y_fit = self.y_fit.as_ref().expect(message);
        (x_fit, errors_fit, coefs_fit, y_fit)
    }

    fn prepare_x_future(&self, h: usize, x: Option<&Array2<f64>>, all_errors: Array1<f64>, coefs: &Array1<f64>) -> Array2<f64> {
        let mut x_future = self.unwrap_x(x, h);

        let errors = lags::create_lags(&all_errors, self.order.q, self.order.s);
        let errors_seasonal = lags::create_lags(&all_errors, self.seasonal_order.q, self.seasonal_order.s);

        let y_lags = Array::zeros((h, self.order.p));
        let y_lags_seasonal = Array::zeros((h, self.seasonal_order.p));

        x_future = self.prepare_x(x_future, errors, errors_seasonal, y_lags, y_lags_seasonal, h as isize);
        if x_future.shape()[1] != coefs.len() {
            panic!("X future must have same number of exogenous variables as used for fit");
        }
        x_future
    }

    pub(super) fn un_difference(&self, y_preds: &mut Array1<f64>) {
        let y_original = self.y_original.as_ref().unwrap();
        *y_preds = difference::un_difference_all(&y_preds, y_original, self.order.d, self.seasonal_order.d, self.seasonal_order.s);
    }
}

impl Model {
    fn unwrap_x(&self, x: Option<&Array2<f64>>, default_length: usize) -> Array2<f64> {
        let x = x.unwrap_or(&Array::zeros((default_length, 0))).to_owned();
        self.check_x_size(default_length, &x);
        x
    }

    fn prepare_x(
        &self,
        x: Array2<f64>,
        errors: Array2<f64>,
        errors_seasonal: Array2<f64>,
        y_lags: Array2<f64>,
        y_lags_seasonal: Array2<f64>,
        size: isize
    ) -> Array2<f64> {
        let intercept: Array2<f64> = Array::ones(size as usize).insert_axis(Axis(1));
        concatenate![
            Axis(1),
            intercept.view(),
            errors.slice(s![-size.., ..]),
            errors_seasonal.slice(s![-size.., ..]),
            y_lags.slice(s![-size.., ..]),
            y_lags_seasonal.slice(s![-size.., ..]),
            x.slice(s![-size.., ..])
        ]
    }

    fn check_x_size(&self, size: usize, x: &Array2<f64>) {
        if x.shape()[0] != size {
            panic!("x is length: {}. It must be length: {}.", x.shape()[0], size);
        }
        match self.x_fit.as_ref() {
            None => (),
            Some(x_fit) => {
                let not_exog = self.order.p + self.seasonal_order.p + self.order.q + self.seasonal_order.q + 1;
                let n_exog = x_fit.shape()[1] - not_exog;
                if x.shape()[1] != n_exog {
                    panic!("x has {} columns. It must should have {}.", x.shape()[1], n_exog);
                }
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
    #[should_panic(expected = "It must be at least")]
    fn test_y_too_small() {
        let model = Model::sarima((2, 1, 3), (1, 1, 1, 7));
        let y = arr1(&[0., 1., 2., 3.]);
        model.prepare_for_fit(&y, None);
    }

    #[test]
    #[should_panic(expected = "It must be at least")]
    fn test_y_too_small_for_x_cols() {
        let model = Model::moving_average(0);
        let y: Array1<f64> = Array::ones(20);
        let x: Array2<f64> = Array::ones((20, 30));
        model.prepare_for_fit(&y, Some(&x));
    }

    #[test]
    #[should_panic(expected = "X future must have same number of exogenous variables as used for fit")]
    fn test_x_future_wrong_cols() {
        
        let h = 10;
        let x_future: Array2<f64> = Array::ones((h, 8));

        let model = Model::moving_average(0);
        let errors: Array1<f64> = Array::zeros(h);
        let n_features = 5;
        let coefs: Array1<f64> = Array::ones(n_features);
        model.prepare_x_future(h, Some(&x_future), errors, &coefs);
    }

    #[test]
    #[should_panic(expected = "must be length")]
    fn test_y_len_not_equal_x_len() {
        let model = Model::sarima((1, 1, 0), (2, 2, 0, 2));
        let y = arr1(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let x: Array2<f64> = arr2(&[[0., 1., 2., 3., 4.], [0., 1., 2., 3., 4.]]).t().to_owned();
        model.prepare_for_fit(&y, Some(&x));
    }
    
    #[test]
    fn test_prepare_for_fit() {
        let model = Model::sarima((1, 1, 3), (2, 0, 2, 7));

        let len = 28;
        let y: Array1<f64> = Array::range(0., len as f64, 1.);
        let x: Array2<f64> = Array::zeros((len, 4));

        let (_, x) = model.prepare_for_fit(&y, Some(&x));

        // len_output should be len_input - order.d - s_order.d*s - max(order.p, s_order.p*s)
        // 29 - 1 - 7 * 0 - max(1, 7 * 2) = 13
        // n_cols = intercept + order.q + s_order.q + order.d + s_order.d + n_x_cols
        let x_result: Array2<f64> = arr2(&[
            // intercept
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            // errors
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            // lags (have been differences so shows ones)
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            // exongenous
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ]).t().to_owned();

        assert_eq!(x, x_result);
    }
}