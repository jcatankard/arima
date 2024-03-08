# arima
Arima time-series forecasting for Python built in Rust

## How run Rust unit testa
`cargo test`

## Build and install in Python virtual environment
`maturin develop`

## How to use Python library

```Python
from arima import Model


X_train, X_test, y_train, y_test = ...

m = Model.sarima(order=ORDER, seasonal_order=SEASONAL_ORDER)
m.fit(y=y_train.values, x=X_train.values)

h = len(X_test)
preds = m.predict(h=h, x=X_test.values)
```
