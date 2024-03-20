pub(crate) mod normal_equation;
use numpy::ndarray::{Array, Array1, Array2, Axis, concatenate, s};
use super::Model;

impl Model {
    pub(super) fn fit_predict_internal(
        &self,
        h: usize,
        mut y: &mut Array1<f64>,
        mut x: &mut Array2<f64>,
        exog: &Array2<f64>
    ) -> (Array1<f64>, Array1<f64>) {

        let (coefs, errors) = self.fit_internal(h, &y, &mut x);

        let new_errors = self.forecast_errors(h, &errors, &exog);
        
        let y_preds = self.predict_internal(h, &mut y, &mut x, &coefs, &new_errors);
        (y_preds, coefs)
    }

    fn fit_internal(&self, h: usize, y: &Array1<f64>, mut x: &mut Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        let (error_start_col, seasonal_error_start_col, seasonal_error_end_col) = self.error_cols();
        let mut coefs: Array1<f64> = Array::zeros(x.shape()[1]);
        let mut errors: Array1<f64> = Array::zeros(y.len());

        let end = y.len() - h;
        for i in 1..end {

            self.move_up(i, &mut x, &errors, error_start_col, seasonal_error_start_col, 1);
            self.move_up(i, &mut x, &errors, seasonal_error_start_col, seasonal_error_end_col, self.seasonal_order.s);

            coefs = normal_equation::solve(x.slice(s![..i, ..]), y.slice(s![..i]));
            let y_pred_i = x.slice(s![i, ..]).dot(&coefs);
            errors[i] = y[i] - y_pred_i;
        }
        (coefs, errors)
    }

    fn predict_internal(&self, h: usize, y: &mut Array1<f64>, mut x: &mut Array2<f64>, coefs: &Array1<f64>, errors: &Array1<f64>) -> Array1<f64> {

        let (lag_start_col, seasonal_lag_start_col, seasonal_lag_end_col) = self.lag_cols();
        let (error_start_col, seasonal_error_start_col, seasonal_error_end_col) = self.error_cols();

        let start = y.len() - h;
        for i in start..y.len() {

            self.move_up(i, &mut x, &y, lag_start_col, seasonal_lag_start_col, 1);
            self.move_up(i, &mut x, &y, seasonal_lag_start_col, seasonal_lag_end_col, self.seasonal_order.s);

            self.move_up(i, &mut x, &errors, error_start_col, seasonal_error_start_col, 1);
            self.move_up(i, &mut x, &errors, seasonal_error_start_col, seasonal_error_end_col, self.seasonal_order.s);

            y[i] = x.slice(s![i, ..]).dot(coefs);
        }
        y.slice(s![-(h as isize)..]).to_owned()
    }

    fn forecast_errors(&self, h: usize, errors: &Array1<f64>, exog: &Array2<f64>) -> Array1<f64> {

        let errors_fit = errors.slice(s![..-(h as isize)]).to_owned();

        let size = exog.shape()[0] - errors.len();  // exog may be longer than errors due to lags
        let exog = exog.slice(s![size.., ..]).to_owned();

        let errors_forecast: Array1<f64> = if self.order.q + self.seasonal_order.q > 0 {

            let exog_future = exog.slice(s![-(h as isize).., ..]).to_owned();
            let exog_fit = exog.slice(s![..-(h as isize), ..]).to_owned();
        
            let mut m = Model::sarima((self.order.p, 0, 0), (self.seasonal_order.p, 0, 0, self.seasonal_order.s));
            m.forecast(&errors_fit, h, Some(&exog_fit), Some(&exog_future))

        } else {
            Array::zeros(h)
        };

        concatenate![Axis(0), errors_fit.view(), errors_forecast.view()]
    }
}

impl Model {
    fn move_up(&self, index: usize, x: &mut Array2<f64>, values: &Array1<f64>, start_col: usize, end_col: usize, s: usize) {
        let s = s as isize;
        for (q, col) in (start_col..end_col).enumerate() {
            let loc = (index as isize) - (q as isize + 1) * s;
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
    use numpy::ndarray::{Array, Array1, Array2, arr2};
    use super::*;

    #[test]
    fn fit_predict_move_up() {
        let len = 5;
        let errors: Array1<f64> = Array::range(5., 5. + len as f64, 1.);
        let n_error_terms = 3;
        let mut x: Array2<f64> = Array::zeros((len, n_error_terms));

        let model = Model::moving_average(n_error_terms);
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
    fn fit_predict_move_up_seasonal() {
        let len = 15;
        let errors: Array1<f64> = Array::range(5., 5. + len as f64, 1.);
        let n_error_terms = 3;
        let mut x: Array2<f64> = Array::zeros((len, n_error_terms));

        let model = Model::sarima((0, 0, 1), (0, 0, 2, 7));
        for i in 0..15 {
            model.move_up(i, &mut x, &errors, 0, 1, 1);
            model.move_up(i, &mut x, &errors, 1, 3, 7);
        }

        let result: Array2<f64> = arr2(&[
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 5.0, 0.0],
            [12.0, 6.0, 0.0],
            [13.0, 7.0, 0.0],
            [14.0, 8.0, 0.0],
            [15.0, 9.0, 0.0],
            [16.0, 10.0, 0.0],
            [17.0, 11.0, 0.0],
            [18.0, 12.0, 5.0]
        ]);

        assert_eq!(result, x);
    }
}