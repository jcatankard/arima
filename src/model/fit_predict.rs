mod normal_equation;
use numpy::ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, s};
use super::Model;


impl Model {
    pub(super) fn fit_internal(&self, y: &Array1<f64>, mut x: &mut Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        let (error_start_col, seasonal_error_start_col, seasonal_error_end_col) = self.error_cols();
        let mut coefs: Array1<f64> = Array::zeros(x.shape()[1]);
        let mut errors: Array1<f64> = Array::zeros(y.len());
        // we need at least as many rows as columns for linear regression solver to work (assuming no regularization)
        let start = x.shape()[1] + 1;
        for i in start..y.len() {

            self.move_up(i, &mut x, &errors, error_start_col, seasonal_error_start_col, 1);
            self.move_up(i, &mut x, &errors, seasonal_error_start_col, seasonal_error_end_col, self.seasonal_order.s);

            coefs = self.solve(y.slice(s![..i]), x.slice(s![..i, ..]));
            let y_pred_i = x.slice(s![i, ..]).dot(&coefs);
            errors[i] = y[i] - y_pred_i;
        }
        (coefs, errors)
    }

    fn solve(&self, _y: ArrayView1<f64>, x: ArrayView2<f64>) -> Array1<f64> {
        // let transpose = x.t();
        // let make_square = transpose.dot(&x);
        // let pseudo_inverse: Array2<f64> = make_square.inv().unwrap().dot(&transpose);
        // pseudo_inverse.dot(&y)
        Array::ones(x.shape()[1])
    }

    pub(super) fn fit_error_model(&mut self, errors: &Array1<f64>, x: &Array2<f64>) {
        if self.order.q + self.seasonal_order.q > 0 {
            self.error_model
                .as_mut()
                .expect("Model should exist if there are error terms")
                .fit(&errors, &Some(&x));
        }
    }
}

impl Model {
    pub(super) fn predict_internal(&self, h: usize, y_preds: &mut Array1<f64>, mut x: Array2<f64>, coefs: &Array1<f64>) {
        let (lag_start_col, seasonal_lag_start_col, seasonal_lag_end_col) = self.lag_cols();

        let start = y_preds.len() - h;
        for i in start..y_preds.len() {

            self.move_up(i, &mut x, &y_preds, lag_start_col, seasonal_lag_start_col, 1);
            self.move_up(i, &mut x, &y_preds, seasonal_lag_start_col, seasonal_lag_end_col, self.seasonal_order.s);

            y_preds[i] = x.slice(s![i, ..]).dot(coefs);
        }
        y_preds.slice_collapse(s![-(h as isize)..]);
    }

    pub(super) fn predict_future_errors(&self, h: usize, x: &Option<&Array2<f64>>) -> Array1<f64> {
        if let Some(m) = self.error_model.as_ref() {m.predict(h, &x)} else {Array::zeros(h)}
    }
}

impl Model {
    fn move_up(&self, index: usize, x: &mut Array2<f64>, values: &Array1<f64>, start_col: usize, end_col: usize, s: usize) {
        let s = s as isize;
        for (q, col) in (start_col..end_col).enumerate() {
            let loc = (index as isize) - (q as isize) * s - 1;
            if loc >= 0 {
                x[[index, col]] = values[loc as usize];
            }
        }
    }
    
    fn error_cols(&self) -> (usize, usize, usize) {
        let error_start_col: usize = 1;  // after intercept
        let seasonal_error_start_col = error_start_col + self.order.q;
        let seasonal_error_end_col = seasonal_error_start_col + self.seasonal_order.q;
        (error_start_col, seasonal_error_start_col, seasonal_error_end_col)
    }

    fn lag_cols(&self) -> (usize, usize, usize) {
        let (_, _, lag_start_col) = self.error_cols();  // after errors
        let seasonal_lag_start_col = lag_start_col + self.order.p;
        let seasonal_lag_end_col = seasonal_lag_start_col + self.seasonal_order.p;
        (lag_start_col, seasonal_lag_start_col, seasonal_lag_end_col)
    }
}


#[cfg(test)]
mod tests {
    // run with "cargo test -- --show-output" to see output
    use numpy::ndarray::{Array, Array1, Array2, arr1, arr2};
    use super::*;

    #[test]
    fn test_move_up() {
        let len = 5;
        let errors: Array1<f64> = Array::range(5., 5. + len as f64, 1.);
        let n_error_terms = 3;
        let mut x: Array2<f64> = Array::zeros((len, n_error_terms));

        let model = Model::moving_average(2);
        model.move_up(0, &mut x, &errors, 0, n_error_terms, 1);
        model.move_up(1, &mut x, &errors, 0, n_error_terms, 1);
        model.move_up(2, &mut x, &errors, 0, n_error_terms, 1);
        model.move_up(3, &mut x, &errors, 0, n_error_terms, 1);

        let result: Array2<f64> = arr2(&[
            // first e-1 is 0 as there is not errors[-1] value
            [0., 5., 6., 7., 0.],
            [0., 0., 5., 6., 0.],
            [0., 0., 0., 5., 0.],
        ]).t().to_owned();

        assert_eq!(result, x);
    }

    #[test]
    fn test_solver() {

        let x: Array2<f64> = arr2(&[
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [0., 5., 6., 7., 0., 4., 3., 2.],
            [0., 0., 5., 6., 0., 9., 8., 7.],
            [0., 0., 0., 5., 0., 1., 2., 3.],
        ]).t().to_owned();

        let coefs: Array1<f64> = arr1(&[-0.5, 0.5, 1.5, 2.5]);
        let y = x.dot(&coefs);

        let model = Model::autoregressive(0);
        let sol = model.solve(y.view(), x.view());

        assert_eq!(coefs, sol);

    }
}