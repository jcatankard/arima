use super::Order;
use numpy::ndarray::{Array1, s};


pub(super) fn difference(y: &Array1<f64>, order: &Order) -> Array1<f64> {
    let mut y = y.to_owned();
    for _ in 0..order.d {
        y = y.slice(s![order.s..]).to_owned() - y.slice(s![..y.len() - order.s]);
    }
    y
}

pub fn _reverse_difference() -> Array1<f64> {
    // https://glarity.app/youtube-summary/entertainment/time-series-talk--arima-model-19761301_10147
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::arr1;
    // run with "cargo test -- --show-output" to see output

    #[test]
    fn test_difference_zero() {
        let y: Array1<f64> = arr1(&[1., 2., 3., 4., 5.]);
        let order = Order {p: 1, d: 0, q: 1, s: 0};
        assert_eq!(difference(&y, &order), y)
    }

    #[test]
    fn test_difference_one() {
        let y: Array1<f64> = arr1(&[1., 2., 3., 4., 5.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1.]);
        let order = Order {p: 1, d: 1, q: 1, s: 1};
        assert_eq!(difference(&y, &order), result)
    }

    #[test]
    fn test_difference_two() {
        let y: Array1<f64> = arr1(&[1., 2., 4., 7., 11., 16., 22.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1., 1.]);
        let order = Order {p: 1, d: 2, q: 1, s: 1};
        assert_eq!(difference(&y, &order), result)
    }

    #[test]
    fn test_difference_three() {
        let y: Array1<f64> = arr1(&[1., 2., 4., 8., 15., 26., 42.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1.]);
        let order = Order {p: 1, d: 3, q: 1, s: 1};
        assert_eq!(difference(&y, &order), result)
    }

    #[test]
    fn test_difference_seasonal_three() {
        let y: Array1<f64> = arr1(&[1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.]);
        let result: Array1<f64> = arr1(&[1., 1., 1., 1., 1., 1., 1., 1., 1.]);
        let order = Order {p: 1, d: 1, q: 1, s: 3};
        assert_eq!(difference(&y, &order), result)
    }

    #[test]
    fn test_difference_one_seasonal_three() {
        let y1: Array1<f64> = arr1(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
        let y2: Array1<f64> = arr1(&[1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.]);
        let y = y1 + y2;

        let order = Order {p: 1, d: 1, q: 1, s: 1};
        let y_diff1 = difference(&y, &order);

        let seasonal_order = Order {p: 1, d: 1, q: 1, s: 3};
        let y_diff2 = difference(&y_diff1, &seasonal_order);

        let result: Array1<f64> = arr1(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(y_diff2, result)
    }
}