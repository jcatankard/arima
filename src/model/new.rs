use super::{Model, Order};


impl Model {
    /// https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average#Variations_and_extensions
    pub fn sarima(order: (usize, usize, usize), seasonal_order: (usize, usize, usize, usize)) -> Self {
        let (p, d, q) = order;
        let order = Order {p, d, q, s: 1};

        let (p, d, q, s) = seasonal_order;
        if s == 1 {
            panic!("It doesn't make sense for periodicity (s) for seasonal_order to be set to 1.");
        }
        let seasonal_order = Order {p, d, q, s};

        Self {order, seasonal_order, y_fit: None, x_fit: None, coefs_fit: None, errors_fit: None}
    }

    /// https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
    pub fn arima(p: usize, d: usize, q: usize) -> Self {
        Self::sarima((p, d, q), (0, 0, 0, 0))
    }

    /// https://en.wikipedia.org/wiki/Autoregressive_moving-average_model
    pub fn arma(p: usize, q: usize) -> Self {
        Self::sarima((p, 0, q), (0, 0, 0, 0))
    }

    /// https://en.wikipedia.org/wiki/Autoregressive_model
    pub fn autoregressive(p: usize) -> Self {
        Self::sarima((p, 0, 0), (0, 0, 0, 0))
    }

    /// https://en.wikipedia.org/wiki/Moving-average_model
    pub fn moving_average(q: usize) -> Self {
        Self::sarima((0, 0, q), (0, 0, 0, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // run with "cargo test -- --show-output" to see output

    #[test]
    #[should_panic(expected = "to be set to 1")]
    fn test_seasonal_s_equal_one() {
        let _model = Model::sarima((1, 2, 3), (4, 5, 6, 1)); 
    }


    #[test]
    fn test_sarima_new() {
        let model = Model::sarima((1, 2, 3), (4, 5, 6, 7));
        assert_eq!(model.order, Order {p: 1, d: 2, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 4, d: 5, q: 6, s: 7});
    }

    #[test]
    fn test_arima_new() {
        let model = Model::arima(1, 2, 3);
        assert_eq!(model.order, Order {p: 1, d: 2, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn test_arma_new() {
        let model = Model::arma(1, 3);
        assert_eq!(model.order, Order {p: 1, d: 0, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn test_ar_new() {
        let model = Model::autoregressive(1);
        assert_eq!(model.order, Order {p: 1, d: 0, q: 0, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn test_ma_new() {
        let model = Model::moving_average(3);
        assert_eq!(model.order, Order {p: 0, d: 0, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

}