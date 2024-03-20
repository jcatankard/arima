use numpy::ndarray::{Array, Array1, ArrayView2, ArrayView1};
use ndarray_linalg::solve::Inverse;


/// X(a, b), Y(a, 1) -> W(b, 1)
pub(crate) fn solve(x: ArrayView2<f64>, y: ArrayView1<f64>) -> Array1<f64> {
    let transpose = x.t();
    let square = transpose.dot(&x);
    
    let square_inverse = match square.inv() {
        Err(_) => {
            let mut penalty = Array::eye(square.shape()[0]) * 1.;
            penalty[[0, 0]] = 0.;  // intercept
            (square + penalty).inv().expect("Should invert.")
        },
        Ok(x) => x,
    };

    let pseudo_inverse = square_inverse.dot(&transpose);
    pseudo_inverse.dot(&y)
}


#[cfg(test)]
mod tests {
    // run with "cargo test -- --show-output" to see output
    use super::*;
    use numpy::ndarray::{arr1, arr2, Array2};

    #[test]
    fn normal_equation_solve() {
        let x: Array2<f64> = arr2(&[
            [1., 2., 3., 4.],
            [1., 7., 6., 7.],
            [1., 1., 1., 1.],
            [1., 3., 2., 4.],
            [1., 7., 6., 7.],
            [1., 1., 1., 1.],
            [1., 3., 2., 4.],
        ]);
        let coefs = arr1(&[-1., 2., 3., 4.]);
        let y = x.dot(&coefs);
        
        assert_eq!(solve(x.view(), y.view()).mapv(|a| a.round()), coefs);

        println!("{:?}", x);
    }
}