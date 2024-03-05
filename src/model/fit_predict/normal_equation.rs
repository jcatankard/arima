use numpy::ndarray::{concatenate, s, Array, Array1, Array2, Axis};

// https://www.mathsisfun.com/algebra/matrix-inverse.html

/// X(a, b), Y(a, c) -> W(b, c)
pub(super) fn solve(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let transpose = x.t();
    let square = transpose.dot(x);
    let square_inverse = inverse(&square);
    let pseudo_inverse = square_inverse.dot(&transpose);
    pseudo_inverse.dot(y)
}

/// X must be square matrix
/// X must be non-singular
fn inverse(x: &Array2<f64>) -> Array2<f64> {
    let d = determinant(&x);
    if d == 0.0 {panic!("X is a singular matrix!")}
    adjoint(&x) / d
}

/// https://www.mathsisfun.com/algebra/matrix-determinant.html
/// Laplace expansion
fn determinant(x: &Array2<f64>) -> f64 {
    if x.len() == 1 {return x[[0, 0]]}
    if x.len() == 4 {return determinant_two_by_two(&x)}
    let mut result: f64 = 0.;
    for i in 0..x.shape()[1] {
        // let mult = (i as f64 % 2.0 - 0.5) * -2.0;  // resolves to +1 or -1
        result += 1.0 * x[[0, i]] * minor(&x, 0, i);
    }
    result
}

fn adjoint(x: &Array2<f64>) -> Array2<f64> {
    cofactor_matrix(&x).t().to_owned()
}

fn cofactor_matrix(x: &Array2<f64>) -> Array2<f64> {
    let (n_rows, n_cols) = (x.shape()[0], x.shape()[1]);
    let mut matrix: Array2<f64> = Array::zeros((n_rows, n_cols));
    for i in 0..n_rows {
        for j in 0..n_cols {
            matrix[[i, j]] = cofactor_element(&x, i, j);
        }
    }
    matrix
}

/// row i, column j
fn cofactor_element(x: &Array2<f64>, i: usize, j: usize) -> f64 {
    minor(&x, i, j).powi((i + j) as i32)
}

/// row 0, column c
fn minor(x: &Array2<f64>, i: usize, j: usize) -> f64 {
    determinant(&sub_matrix(&x, i, j))
}


/// row i, column j
fn sub_matrix(x: &Array2<f64>, i: usize, j: usize) -> Array2<f64> {
    let top_left = x.slice(s![..i, ..j]);
    let bot_left = x.slice(s![(i + 1).., ..j]);
    let top_right = x.slice(s![..i, (j + 1)..]);
    let bot_right = x.slice(s![(i + 1).., (j + 1)..]);
    let left = concatenate![Axis(0), top_left, bot_left];
    let right = concatenate![Axis(0), top_right, bot_right];
    concatenate![Axis(1), left, right]
}

fn sub_matrix1(x: &Array2<f64>, i: usize, j: usize) -> Array2<f64> {
    let mut s: Array2<f64> = Array::zeros((x.shape()[0] - 1, x.shape()[1] - 1));
    s.slice_mut(s![..i, ..j]).assign(&x.slice(s![..i, ..j]));
    s.slice_mut(s![i.., ..j]).assign(&x.slice(s![(i + 1).., ..j]));
    s.slice_mut(s![..i, j..]).assign(&x.slice(s![..i, (j + 1)..]));
    s.slice_mut(s![i.., j..]).assign(&x.slice(s![(i + 1).., (j + 1)..]));
    s
}

/// X must be a 2x2 matrix
fn determinant_two_by_two(x: &Array2<f64>) -> f64 {
    let (a, b, c, d) = (x[[0, 0]], x[[0, 1]], x[[1, 0]], x[[1, 1]]);
    a * d - b * c
}

#[cfg(test)]
mod tests {
    // run with "cargo test -- --show-output" to see output
    use super::*;
    use numpy::ndarray::{Array, arr2};

    #[test]
    #[should_panic(expected = "X is a singular matrix!")]
    fn test_panic_singular() {
        let x: Array2<f64> = arr2(&[
            [1., 2.],
            [3., 6.],
        ]);
        inverse(&x);
    }

    #[test]
    fn test_two_by_two() {
        let (a, b, c, d) = (1., 2., 3., 4.);
        let x: Array2<f64> = arr2(&[
            [a, b],
            [c, d],
        ]);
        let x_adj: Array2<f64> = arr2(&[
            [d, -b],
            [-c, a],
        ]);
        let x_determ = a * d - b * c;
        let x_inv = (1. / x_determ) * x_adj;
        assert_eq!(inverse(&x), x_inv);
    }

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
        assert_eq!(x.dot(&inverse(&x)), identity);
    }

    #[test]
    fn test_determinant_two_by_two() {
        let x: Array2<f64> = arr2(&[
            [4., 6.],
            [3., 8.],
        ]);
        assert_eq!(determinant_two_by_two(&x), 14.)
    }

    #[test]
    fn test_determinant_three_by_three() {
        let x: Array2<f64> = arr2(&[
            [6., 1., 1.],
            [4., -2., 5.],
            [2., 8., 7.],
        ]);
        assert_eq!(determinant(&x), -306.)
    }

    #[test]
    fn test_submatrix() {
        let x: Array2<f64> = arr2(&[
            [1., 2., 3., 4.],
            [6., 7., 6., 7.],
            [1., 1., 1., 1.],
            [4., 3., 2., 4.],
        ]);
        let y: Array2<f64> = arr2(&[
            [1., 2., 4.],
            [6., 7., 7.],
            [4., 3., 4.],
        ]);
        assert_eq!(sub_matrix1(&x, 2, 2), y);
    }

    #[test]
    fn test_submatrix1() {
        let x: Array2<f64> = arr2(&[
            [1., 2., 3., 4.],
            [6., 7., 6., 7.],
            [1., 1., 1., 1.],
            [4., 3., 2., 4.],
        ]);
        assert_eq!(sub_matrix(&x, 2, 2), sub_matrix1(&x, 2, 2));
    }

    #[test]
    fn test_minor() {
        let x: Array2<f64> = arr2(&[
            [1., 2., 4.],
            [6., 7., 7.],
            [4., 3., 4.],
        ]);
        // for 1, 1
        // sub matrix
        // [1., 4.]
        // [4., 4.]
        // 1 * 4 - 4 * 4 = -12
        assert_eq!(minor(&x, 1, 1), -12.);
    }

    #[test]
    fn test_cofactor_element() {
        let x: Array2<f64> = arr2(&[
            [1., 2., 4.],
            [6., 7., 7.],
            [4., 3., 4.],
        ]);
        // for 1, 1 is minor at 1, 1 ^ (1 + 1)
        assert_eq!(cofactor_element(&x, 1, 1), 144.);
    }

    #[test]
    fn test_cofactor_matrix() {
        let x: Array2<f64> = arr2(&[
            [1., 2., 4.],
            [6., 7., 7.],
            [4., 3., 4.],
        ]);
        let minors: Array2<f64> = arr2(&[
            [7., -4., -10.],
            [-4., -12., -5.],
            [-14., -17., -5.],
        ]);
        let pow: Array2<f64> = arr2(&[
            [0., 1., 2.],
            [1., 2., 3.],
            [2., 3., 4.],
        ]);
        let mut cofactor_result: Array2<f64> = Array::zeros((x.shape()[0], x.shape()[1]));
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                cofactor_result[[i, j]] = minors[[i, j]].powf(pow[[i, j]]);
            }
        }
        assert_eq!(cofactor_matrix(&x), cofactor_result);
    }

    #[test]
    fn test_adjoint() {
        let x: Array2<f64> = arr2(&[
            [1., 2., 4.],
            [6., 7., 7.],
            [4., 3., 4.],
        ]);
        // assert_eq!(inverse(&x), x_inv);
    }
}