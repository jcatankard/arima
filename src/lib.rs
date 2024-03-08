//! # ARIMA
//! The purpose of this crate is to create a Rust implemenation of the following forecasters:
//! - ARIMA
//! - SARIMA
//! - ARIMAX / SARIMAX
//! - ARMA
//! - Autoregressive
//! - Moving average
//! 
//! This crate also serves as a Python extension module.
//! 
//! Check out [Wikipedia](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) for more information.
//! 

mod model;
pub use model::Model;

use numpy::ndarray::{Array, Array2};
use numpy::{IntoPyArray, PyArray1, PyArrayLike1, PyArrayLike2};
use pyo3::{Python, PyResult, pymethods, pymodule};
use pyo3::types::{PyModule, PyType};

fn unwrap_x(x: Option<PyArrayLike2<f64>>, default_length: usize) -> Array2<f64> {
    match x {
        None => Array::zeros((default_length, 0)),
        Some(a) => a.as_array().to_owned()
    }
}

#[pymethods]
impl Model {
    #[pyo3(name = "fit")]
    fn py_fit<'py>(&mut self, y: PyArrayLike1<'py, f64>, x: Option<PyArrayLike2<'py, f64>>) {
        self.fit(&y.as_array().to_owned(), Some(&unwrap_x(x, y.len())))
    }
    #[pyo3(name = "predict")]
    fn py_predict<'py>(&self, py: Python<'py>, h: usize, x: Option<PyArrayLike2<'py, f64>>
) -> &'py PyArray1<f64> {
        self.predict(h, Some(&unwrap_x(x, h))).into_pyarray(py)
    }

    #[pyo3(name = "forecast")]
    fn py_forecast<'py>(&mut self, py: Python<'py>, y: PyArrayLike1<'py, f64>, h: usize, x: Option<PyArrayLike2<'py, f64>>, x_future: Option<PyArrayLike2<'py, f64>>
) -> &'py PyArray1<f64> {
        self.forecast(&y.as_array().to_owned(), h, Some(&unwrap_x(x, y.len())), Some(&unwrap_x(x_future, h))).into_pyarray(py)
    }

    #[pyo3(name = "fit_predict")]
    fn py_fit_predict<'py>(&mut self, py: Python<'py>, y: PyArrayLike1<'py, f64>, h: usize, x: Option<PyArrayLike2<'py, f64>>, x_future: Option<PyArrayLike2<'py, f64>>
) -> &'py PyArray1<f64> {
        self.fit_predict(&y.as_array().to_owned(), h, Some(&unwrap_x(x, y.len())), Some(&unwrap_x(x_future, h))).into_pyarray(py)
    }

    // https://pyo3.rs/v0.20.3/class#class-methods
    #[classmethod]
    #[pyo3(name = "sarima")]
    fn py_sarima(_cls: &PyType, order: (usize, usize, usize), seasonal_order: (usize, usize, usize, usize)) -> PyResult<Self> {
        Ok(Self::sarima(order, seasonal_order))
    }

    #[classmethod]
    #[pyo3(name = "arima")]
    fn py_arima(_cls: &PyType, p: usize, d: usize, q: usize) -> PyResult<Self> {
        Ok(Self::arima(p, d, q))
    }

    #[classmethod]
    #[pyo3(name = "arma")]
    fn py_arma(_cls: &PyType, p: usize, q: usize) -> PyResult<Self> {
        Ok(Self::arma(p, q))
    }

    #[classmethod]
    #[pyo3(name = "autoregressive")]
    fn py_autoregressive(_cls: &PyType, p: usize) -> PyResult<Self> {
        Ok(Self::autoregressive(p))
    }

    #[classmethod]
    #[pyo3(name = "moving_average")]
    fn py_moving_average(_cls: &PyType, q: usize) -> PyResult<Self> {
        Ok(Self::moving_average(q))
    }

//     #[getter]
//     fn coefs<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
//         Ok(self.coefs_fit.to_owned().unwrap().into_pyarray(py))
//     }
}


#[pymodule]
#[pyo3(name = "arima")]
fn arima<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    // https://pyo3.rs/v0.20.3/class    
    m.add_class::<Model>()?;
    Ok(())
}
