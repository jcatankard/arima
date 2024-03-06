use numpy::ndarray::{Array1, Axis, s, concatenate};

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
    let y_last_len = y_last.len();
    if s <= y_last_len {
        let mut y_temp = concatenate![Axis(0), y_last.view(), y_differenced.view()];
        for i in y_last_len..y_temp.len() {
            y_temp[i] = y_temp[i - s];
        }
        return y_temp.slice(s![y_last_len..]).to_owned();
    }
    y_differenced.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use numpy::ndarray::{Array, arr1, s};
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
    fn test_undifference_one_degrees() {
        let mut model = Model::sarima((0, 1, 0), (0, 0, 0, 0));

        let y: Array1<f64> = Array::range(0., 25., 1.);

        let cutoff = 14;
        let y_train = y.slice(s![..cutoff]).to_owned();
        model.fit(&y_train, None);

        let y_future = y.slice(s![cutoff..]).to_owned();

        let mut y_preds = model.difference_y(&y).slice(s![cutoff..]).to_owned();
        model.un_difference(&mut y_preds);
        
        assert_eq!(y_future, y_preds);
    }

    
    #[test]
    fn test_undifference_two_degrees() {
        let mut model = Model::sarima((0, 2, 0), (0, 0, 0, 0));

        let y: Array1<f64> = arr1(&[1., 2., 4., 7., 11., 16., 22., 29., 37., 46., 56., 67., 79., 92., 106., 121., 137., 154., 172.]);

        let cutoff = 14;
        let y_train = y.slice(s![..cutoff]).to_owned();
        model.fit(&y_train, None);

        let y_future = y.slice(s![cutoff..]).to_owned();

        let mut y_preds = model.difference_y(&y).slice(s![cutoff..]).to_owned();
        model.un_difference(&mut y_preds);
        
        assert_eq!(y_future, y_preds);
    }
}