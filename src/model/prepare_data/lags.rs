use numpy::ndarray::{Array1, Array2, Array, s};

pub(super) fn create_lags(y: &Array1<f64>, p: usize, s: usize) -> Array2<f64> {
    let len = y.len() - p * s;
    let mut y_lags: Array2<f64> = Array::zeros((len, p));
    for i in 0..p {
        let start = s * (p - i - 1);
        let values = y.slice(s![start..start + len]);
        y_lags.slice_mut(s![.., i]).assign(&values);
    }
    y_lags
}


#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{arr1, arr2};

    #[test]
    fn lags_zero() {
        let y = arr1(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let y_lags = create_lags(&y, 0, 0);
        let result: Array2<f64> = Array::zeros((y.len(), 0));
        assert_eq!(result, y_lags);
    }

    #[test]
    fn lags_one() {
        let y = arr1(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let y_lags = create_lags(&y, 1, 1);

        let result = arr2(&[[0., 1., 2., 3., 4., 5., 6., 7., 8.]]);
        assert_eq!(result.t(), y_lags);
    }

    #[test]
    fn lags_two() {
        let y = arr1(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let y_lags = create_lags(&y, 2, 1);

        let result = arr2(&[
            [1., 2., 3., 4., 5., 6., 7., 8.],
            [0., 1., 2., 3., 4., 5., 6., 7.]
        ]);
        assert_eq!(result.t(), y_lags);
    }

    #[test]
    fn lags_three() {
        let y = arr1(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let y_lags = create_lags(&y, 3, 1);
        let result = arr2(&[
            [2., 3., 4., 5., 6., 7., 8.],
            [1., 2., 3., 4., 5., 6., 7.],
            [0., 1., 2., 3., 4., 5., 6.]
        ]);
        assert_eq!(result.t(), y_lags);
    }

    #[test]
    fn lags_two_seasonal_two() {
        let y = arr1(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let y_lags = create_lags(&y, 2, 2);
        let result = arr2(&[
            [2., 3., 4., 5., 6., 7.],
            [0., 1., 2., 3., 4., 5.]
        ]);
        println!("{:?}", result.t());
        println!("{:?}", y_lags);
        assert_eq!(result.t(), y_lags);
    }

    #[test]
    fn lags_three_seasonal_three() {
        let y = arr1(&[10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.]);
        let y_lags = create_lags(&y, 3, 3);
        let result = arr2(&[
            [16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.],
            [13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.],
            [10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]
        ]);
        assert_eq!(result.t(), y_lags);
    }
}