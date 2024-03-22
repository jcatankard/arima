# ARIMA
Arima time-series forecasting for Python built in Rust.

[Medium blog post](https://medium.com/@jcatankard_76170/sarimax-in-rust-d71488afd10b).

## Run Rust unit tests
`cargo test`

## Create Python virtual environment
`python -m venv <virtual-environment-name>`

To activate:

`<virtual-environment-name>/Scripts/activate`

## Build and install in Python virtual environment
`maturin develop` to build the wheels and install in Python virtual environment. [Docs](https://www.maturin.rs/local_development) for more info.

## How to use Python library

### 1. SARIMA
```Python
from arima import Model


X_train, X_test, y_train, y_test = ...

m = Model.sarima(order=(3, 1, 3), seasonal_order=(1, 0, 1))
m.fit(y=y_train.values, x=X_train.values)

h = len(X_test)
preds = m.predict(h=h, x=X_test.values)
```

### 2. ARIMA
```Python
from arima import Model


X_train, X_test, y_train, y_test = ...

m = Model.arima(p=2, d=1, q=1)
m.fit(y=y_train.values, x=X_train.values)

h = len(X_test)
preds = m.predict(h=h, x=X_test.values)
```

### 3. ARMA
```Python
from arima import Model


X_train, X_test, y_train, y_test = ...

m = Model.arma(p=2, d=1, q=1)
m.fit(y=y_train.values, x=X_train.values)

h = len(X_test)
preds = m.predict(h=h, x=X_test.values)
```

### 4. Autoregressive
```Python
from arima import Model


X_train, X_test, y_train, y_test = ...

m = Model.autoregressive(3)
m.fit(y=y_train.values, x=X_train.values)

h = len(X_test)
preds = m.predict(h=h, x=X_test.values)
```

### 5. Moving Average
```Python
from arima import Model


X_train, X_test, y_train, y_test = ...

m = Model.moving_average(2)
m.fit(y=y_train.values, x=X_train.values)

h = len(X_test)
preds = m.predict(h=h, x=X_test.values)
```