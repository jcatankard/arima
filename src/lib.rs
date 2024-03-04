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

pub use model::Model;

use pyo3::prelude::*;
mod model;


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn arima(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
