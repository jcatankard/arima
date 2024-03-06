extern crate intel_mkl_src;
use numpy::ndarray::{Array1, Array2};
use ndarray_linalg::solve::Inverse;


/// X(a, b), Y(a, c) -> W(b, c)
pub(super) fn solve(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let transpose = x.t();
    let square = transpose.dot(x);
    // let square_inverse = inverse(&square);
    let square_inverse = square.inv().expect("Square matrix should invert");
    let pseudo_inverse = square_inverse.dot(&transpose);
    pseudo_inverse.dot(y)
}


#[cfg(test)]
mod tests {
    // run with "cargo test -- --show-output" to see output
    use super::*;
    use numpy::ndarray::{Array, arr2};

    #[test]
    fn test_identity() {
        let x: Array2<f64> = arr2(&[
            [1., 2., 3., 4.],
            [6., 7., 6., 7.],
            [1., 1., 1., 1.],
            [4., 3., 2., 4.],
        ]);
        let mut identity: Array2<f64> = Array::zeros((x.shape()[0], x.shape()[1]));
        for i in 0..x.shape()[0] {
            identity[[i, i]] = 1.0;
        }
        // assert_eq!(inverse(&x).dot(&x), identity);
        assert_eq!(x.dot(&x.inv().expect("")), identity);
    }
}