use numpy::ndarray::{Array1, s};

/// d: degree of differences
/// s: periodicity
pub(super) fn difference(y: &Array1<f64>, d: usize, s: usize) -> Array1<f64> {
    let mut y = y.to_owned();
    for _ in 0..d {
        y = y.slice(s![s..]).to_owned() - y.slice(s![..y.len() - s]);
    }
    y
}

/// differences back to the previous level therefore if d = n, this operation needs running n times.
/// y_differenced: y_pred to be differenced from d-1 to d
/// y_last: y_fit at level d
/// s: periodicity
pub(super) fn un_difference(y_differenced: &Array1<f64>, y_last: &Array1<f64>, s: usize) -> Array1<f64> {

    let mut y_new = y_differenced.to_owned();
    for i in 0..s {
        y_new[i] += y_last[y_last.len() - 1 * s + i];
    }
    
    for i in s..y_new.len() {
        y_new[i] += y_new[i - s];
    }
    y_new
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{arr1, s};
    // run with "cargo test -- --show-output" to see output

    #[test]
    fn test_difference_zero() {
        let y: Array1<f64> = arr1(&[1., 2., 3., 4., 5.]);
        assert_eq!(difference(&y, 0, 0), y)
    }

    #[test]
    fn test_difference_one() {
        let y: Array1<f64> = arr1(&[1., 2., 3., 4., 5.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1.]);
        assert_eq!(difference(&y, 1, 1), result)
    }

    #[test]
    fn test_difference_two() {
        let y: Array1<f64> = arr1(&[1., 2., 4., 7., 11., 16., 22.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1., 1.]);
        assert_eq!(difference(&y, 2, 1), result)
    }

    #[test]
    fn test_difference_three() {
        let y: Array1<f64> = arr1(&[1., 2., 4., 8., 15., 26., 42.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1.]);
        assert_eq!(difference(&y, 3, 1), result)
    }

    #[test]
    fn test_difference_seasonal_three() {
        let y: Array1<f64> = arr1(&[1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1., 1., 1., 1., 1., 1.]);
        assert_eq!(difference(&y, 1, 3), result)
    }

    #[test]
    fn test_difference_one_seasonal_three() {
        let y1: Array1<f64> = arr1(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
        let y2: Array1<f64> = arr1(&[1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.]);
        let y = y1 + y2;

        let y_diff1 = difference(&y, 1, 1);

        let y_diff2 = difference(&y_diff1, 1, 3);

        let result: Array1<f64> = arr1(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(y_diff2, result)
    }

    #[test]
    fn test_undifference() {
        let y: Array1<f64> = arr1(&[1., 2., 4., 7., 11., 16., 22., 29., 37., 46., 56., 67., 79., 92.]);
        let s = 1;
        let y_diff = difference(&y, 1, s);
        let y_undiff = un_difference(&y_diff, &y.slice(s![..s]).to_owned(), s);
        assert_eq!(y.slice(s![s..]), y_undiff);
    }

    
    #[test]
    fn test_undifference_seasonal() {
        let y: Array1<f64> = arr1(&[1., 2., 4., 7., 11., 16., 22., 29., 37., 46., 56., 67., 79., 92.]);
        let s = 3;
        let y_diff = difference(&y, 1, s);
        let y_undiff = un_difference(&y_diff, &y.slice(s![..s]).to_owned(), s);
        assert_eq!(y.slice(s![s..]), y_undiff);
    }
}