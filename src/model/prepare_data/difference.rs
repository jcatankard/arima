use numpy::ndarray::{Array1, Array2, Axis, s, concatenate};

/// a: time series to difference
/// d: degree of differences
/// s: periodicity
fn diff2d(a: &Array2<f64>, d: usize, s: usize) -> Array2<f64> {
    let mut a = a.to_owned();
    for _ in 0..d {
        a = a.slice(s![s.., ..]).to_owned() - a.slice(s![..a.shape()[0] - s, ..]);
    }
    a
}

/// a: time series to difference
/// d: degree of differences
/// s_d: degree od seasonal differences
/// s: peridicity of season
pub(super) fn diff_all2d(a: &Array2<f64>, d: usize, s_d: usize, s: usize) -> Array2<f64> {
    diff2d(&diff2d(&a, d, 1), s_d, s)
}


/// y: time series to difference
/// d: degree of differences
/// s: periodicity
fn diff1d(y: &Array1<f64>, d: usize, s: usize) -> Array1<f64> {
    let mut y = y.to_owned();
    for _ in 0..d {
        y = y.slice(s![s..]).to_owned() - y.slice(s![..y.len() - s]);
    }
    y
}

// let y = y.to_owned().insert_axis(Axis(1));
// diff2d(&y, d, s).slice(s![.., 0]).to_owned()

/// y: time series to difference
/// d: degree of differences
/// s_d: degree od seasonal differences
/// s: peridicity of season
pub(super) fn diff_all1d(y: &Array1<f64>, d: usize, s_d: usize, s: usize) -> Array1<f64> {
    diff1d(&diff1d(&y, d, 1), s_d, s)
}

/// differences back to the previous level therefore if d = n, this operation needs running n times.
/// y: time series to difference
/// y_last: y_fit at level d
/// s: periodicity
fn integrate(y: &Array1<f64>, y_last: &Array1<f64>, s: usize) -> Array1<f64> {
    let mut y_integrated = concatenate![Axis(0), y_last.view(), y.view()];
    for i in y_last.len()..y_integrated.len() {
        y_integrated[i] += y_integrated[i - s];
    }
    y_integrated.slice(s![y_last.len()..]).to_owned()
}

/// y_preds: predictions before being integrated
/// y_original: the original values of y for used for fitting before being differenced
/// d: degree of differences
/// s_d: degree od seasonal differences
/// s: peridicity of season
pub(super) fn integrate_all(y_preds: &Array1<f64>, y_original: &Array1<f64>, d: usize, s_d: usize, s: usize) -> Array1<f64> {

    let mut y_integrated = y_preds.to_owned();

    for i in (0..s_d).rev() {
        let y_last = diff_all1d(&y_original, d, i, s);
        y_integrated = integrate(&y_integrated, &y_last, s);
    }

    for i in (0..d).rev() {
        let y_last = diff1d(&y_original, i, 1);
        y_integrated = integrate(&y_integrated, &y_last, 1);
    }
    y_integrated
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{Array, arr1, s};
    // run with "cargo test -- --show-output" to see output

    #[test]
    fn difference_difference_zero() {
        let y: Array1<f64> = arr1(&[1., 2., 3., 4., 5.]);
        assert_eq!(diff1d(&y, 0, 0), y)
    }

    #[test]
    fn difference_difference_one() {
        let y: Array1<f64> = arr1(&[1., 2., 3., 4., 5.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1.]);
        assert_eq!(diff1d(&y, 1, 1), result)
    }

    #[test]
    fn difference_difference_two() {
        let y: Array1<f64> = arr1(&[1., 2., 4., 7., 11., 16., 22.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1., 1.]);
        assert_eq!(diff1d(&y, 2, 1), result)
    }

    #[test]
    fn difference_difference_three() {
        let y: Array1<f64> = arr1(&[1., 2., 4., 8., 15., 26., 42.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1.]);
        assert_eq!(diff1d(&y, 3, 1), result)
    }

    #[test]
    fn difference_seasonal_one_degrees() {

        let (d, s_d, s) = (0, 1, 7);

        let y: Array1<f64> = arr1(&[7., 6., 4., 3., 4., 5., 6., 7., 6., 4., 3., 4., 5., 6., 7., 6., 4., 3., 4., 5., 6., 7., 6., 4., 3., 4., 5., 6.]);

        let y_diff = diff_all1d(&y, d, s_d, s);
        let result: Array1<f64> = Array::zeros(y.len() - s);
        assert_eq!(result, y_diff);
    }

    #[test]
    fn difference_difference_seasonal_three() {
        let y: Array1<f64> = arr1(&[1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1., 1., 1., 1., 1., 1.]);
        assert_eq!(diff1d(&y, 1, 3), result)
    }

    #[test]
    fn difference_difference_one_seasonal_three() {
        let y1: Array1<f64> = arr1(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
        let y2: Array1<f64> = arr1(&[1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.]);
        let y = y1 + y2;

        let y_diff1 = diff1d(&y, 1, 1);
        let y_diff2 = diff1d(&y_diff1, 1, 3);

        let result: Array1<f64> = arr1(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(y_diff2, result)
    }

    #[test]
    fn difference_integrate_one_degree() {

        let (d, s_d, s) = (1, 0, 0);

        let y: Array1<f64> = Array::range(0., 100., 2.);

        let cutoff = 14;
        let from_end = (y.len() - cutoff) as isize;

        let y_train = y.slice(s![..cutoff]).to_owned();
        let y_future = y.slice(s![cutoff..]).to_owned();

        let mut y_preds = diff_all1d(&y, d, s_d, s).slice(s![-from_end..]).to_owned();
        y_preds = integrate_all(&y_preds, &y_train, d, s_d, s);
        
        assert_eq!(y_future, y_preds);
    }
    
    #[test]
    fn difference_integrate_two_degrees() {
        let (d, s_d, s) = (2, 0, 0);

        let y: Array1<f64> = arr1(&[1., 2., 4., 7., 11., 16., 22., 29., 37., 46., 56., 67., 79., 92., 106., 121., 137., 154., 172.]);

        let cutoff = 14;
        let from_end = (y.len() - cutoff) as isize;

        let y_train = y.slice(s![..cutoff]).to_owned();
        let y_future = y.slice(s![cutoff..]).to_owned();

        let mut y_preds = diff_all1d(&y, d, s_d, s).slice(s![-from_end..]).to_owned();
        y_preds = integrate_all(&y_preds, &y_train, d, s_d, s);
        
        assert_eq!(y_future, y_preds);
    }

    #[test]
    fn difference_integrate_two_one_degrees() {

        let (d, s_d, s) = (2, 1, 2);

        let y1: Array1<f64> = arr1(&[1., 2., 4., 7., 11., 16., 22., 29., 37., 46., 56., 67., 79., 92., 106., 121., 137., 154., 172.]);
        let y2: Array1<f64> = arr1(&[1., 4., 1., 4., 1., 4., 1., 4., 1., 4., 1., 4., 1., 4., 1., 4., 1., 4., 1.]);
        let y = y1 + y2;

        let cutoff = 14;
        let from_end = (y.len() - cutoff) as isize;
        let y_train = y.slice(s![..cutoff]).to_owned();

        let y_future = y.slice(s![cutoff..]).to_owned();

        let mut y_preds = diff_all1d(&y, d, s_d, s).slice(s![-from_end..]).to_owned();
        y_preds = integrate_all(&y_preds, &y_train, d, s_d, s);
        
        assert_eq!(y_future, y_preds);
    }

    #[test]
    fn difference_integrate_seasonal_one_degrees() {

        let (d, s_d, s) = (0, 1, 7);

        let y: Array1<f64> = arr1(&[7., 6., 4., 3., 4., 5., 6., 7., 6., 4., 3., 4., 5., 6., 7., 6., 4., 3., 4., 5., 6., 7., 6., 4., 3., 4., 5., 6.]);

        let cutoff = 14;
        let from_end = (y.len() - cutoff) as isize;
        let y_train = y.slice(s![..cutoff]).to_owned();

        let y_future = y.slice(s![cutoff..]).to_owned();

        let mut y_preds = diff_all1d(&y, d, s_d, s).slice(s![-from_end..]).to_owned();
        y_preds = integrate_all(&y_preds, &y_train, d, s_d, s);
        
        assert_eq!(y_future, y_preds);
    }
}