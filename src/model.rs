mod prepare_data;
mod fit_predict;

use numpy::ndarray::{Array1, Array2};
use pyo3::pyclass;


#[derive(Debug)]
#[pyclass(name = "Model", module = "arima")]
pub struct Model {
    // order: (AR(p), I(d), MA(q), 1)
    // seasonal_order: (AR(p), I(d), MA(q), s)
    // x_fit: data used for fitting incl. exongenous variables, lags and error terms
    // coefs_fit: last coefficients from fitting
    // errors_fit: y - ŷ
    // error_model: forecasting future errors for MA models
    order: Order,
    seasonal_order: Order,
    endog_fit: Option<Array1<f64>>,
    exog_fit: Option<Array2<f64>>,
    pub coefs: Option<Array1<f64>>
}

/// p: AR (auto regressive) terms
/// d: I (integrated) terms
/// q: MA (moving average) terms
/// s: periodicity
#[derive(PartialEq, Debug)]
struct Order {
    p: usize,
    d: usize,
    q: usize,
    s: usize
}

/// # Train and forecast
/// 
impl Model {
    /// - y: timeseries
    /// - x: exogenous variables, same length as y
    pub fn fit(&mut self, y: &Array1<f64>, x: Option<&Array2<f64>>) {
        self.endog_fit = Some(y.to_owned());
        self.exog_fit = Some(self.unwrap_x(x, y.len()));
    }

    /// - h: horizons to forecast
    /// - x: future exongenous variables, same length as h
    /// 
    /// returns predictions for h horizons
    pub fn predict(&mut self, h: usize, x: Option<&Array2<f64>>) -> Array1<f64> {

        let exog_fit = self.exog_fit.as_ref().expect("Model must be fit before predict");
        let exog_future = self.unwrap_x(x, h);
        let endog_fit = self.endog_fit.as_ref().expect("Model must be fit before predict");

        let (exog_diff, endog_diff) = self.difference_xy(exog_fit, &exog_future, endog_fit, h);
        let (mut x, mut y) = self.prepare_xy(&exog_diff, &endog_diff);
        
        let (y_preds, coefs) = self.fit_predict_internal(h, &mut y, &mut x, &exog_diff);
        self.coefs = Some(coefs);
        self.integrate_predictions(&y_preds, &endog_fit)
    }

    /// - y: timeseries
    /// - h: horizons to forecast
    /// - x: exogenous variables, same length as y
    /// - x_future: future exongenous variables, same length as h
    /// 
    /// returns predictions for h horizons
    pub fn forecast(&mut self, y: &Array1<f64>, h: usize, x: Option<&Array2<f64>>, x_future: Option<&Array2<f64>>) -> Array1<f64> {
        self.fit(&y, x);
        self.predict(h, x_future)
    }

    /// - y: timeseries
    /// - h: horizons to forecast
    /// - x: exogenous variables, same length as y
    /// - x_future: future exongenous variables, same length as h
    /// 
    /// returns predictions for h horizons
    pub fn fit_predict(&mut self, y: &Array1<f64>, h: usize, x: Option<&Array2<f64>>, x_future: Option<&Array2<f64>>) -> Array1<f64> {
        self.forecast(&y, h, x, x_future)
    }

    /// Create a [SARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average#Variations_and_extensions) model.
    /// - order: (p, d, q)
    ///     - p: AR(p) auto regressive terms
    ///     - d: I(d) integrated terms
    ///     - q: MA(q) moving average terms
    /// - seasonal_order: (P, D, Q, s)
    ///     - P: AR(P) auto regressive terms
    ///     - D: I(D) integrated terms
    ///     - Q: MA(Q) moving average terms
    ///     - s: periodicity
    /// 
    pub fn sarima(order: (usize, usize, usize), seasonal_order: (usize, usize, usize, usize)) -> Self {
        let (p, d, q) = order;
        let order = Order {p, d, q, s: 1};

        let (p, d, q, s) = seasonal_order;
        if s == 1 {panic!("It doesn't make sense for periodicity (s) to be set to 1.")}
        let seasonal_order = Order {p, d, q, s};

        Self {order, seasonal_order, endog_fit: None, exog_fit: None, coefs: None}
    }

    /// Create an [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) model
    /// - p: AR(p) auto regressive terms
    /// - d: I(d) integrated terms
    /// - q: MA(q) moving average terms
    /// 

    pub fn arima(p: usize, d: usize, q: usize) -> Self {
        Self::sarima((p, d, q), (0, 0, 0, 0))
    }

    /// Create an [ARMA](https://en.wikipedia.org/wiki/Autoregressive_moving-average_model) model
    /// - p: AR(p) auto regressive terms
    /// - q: MA(q) moving average terms
    ///

    pub fn arma(p: usize, q: usize) -> Self {
        Self::sarima((p, 0, q), (0, 0, 0, 0))
    }

    /// Create an [Autoregressive](https://en.wikipedia.org/wiki/Autoregressive_model) model
    /// - p: AR(p) auto regressive terms

    pub fn autoregressive(p: usize) -> Self {
        Self::sarima((p, 0, 0), (0, 0, 0, 0))
    }

    /// Create a [Moving averages](https://en.wikipedia.org/wiki/Moving-average_model) model
    /// - q: MA(q) moving average terms

    pub fn moving_average(q: usize) -> Self {
        Self::sarima((0, 0, q), (0, 0, 0, 0))
    }
}


#[cfg(test)]
mod tests {
    // run with "cargo test -- --show-output" to see output
    use numpy::ndarray::{Array, Array1, arr1, s};
    use super::*;

    #[test]
    fn model_autoregressive() {

        let (cons, lag1, lag2) = (100., 0.5, -0.25);

        let mut y: Array1<f64> = Array::zeros(200) + cons;
        y[0] = 150.;
        y[1] = 50.;

        for i in 2..y.len() {
            y[i] += y[i - 1] * lag1 + y[i - 2] * lag2;
        }

        let y_train = y.slice(s![..180]).to_owned();
        let mut y_test = y.slice(s![180..]).to_owned();
       
        let mut model = Model::autoregressive(2);
        model.fit(&y_train, None);

        let coefs = model.coefs.as_ref().unwrap().mapv(|x| (100. * x).round() / 100.);

        assert_eq!(arr1(&[cons, lag1, lag2]), coefs);

        let y_preds = model.predict(20, None).mapv(|x| (100. * x).round() / 100.);
        y_test = y_test.mapv(|x| (100. * x).round() / 100.);
        assert_eq!(y_test, y_preds);
    }

    #[test]
    fn model_seasonal_ar() {

        let (cons, lag1, lag2, lag_s, s) = (60., 0.45, -0.35, 0.25, 7);

        let mut y: Array1<f64> = Array::zeros(100) + cons;

        for i in s..y.len() {
            y[i] += y[i - 1] * lag1 + y[i - 2] * lag2 + y[i - s] * lag_s;
        }

        let y_train = y.slice(s![..80]).to_owned();
        let mut y_test = y.slice(s![80..]).to_owned();
       
        let mut model = Model::sarima((2, 0, 0), (1, 0, 0, s));
        model.fit(&y_train, None);

        let coefs = model.coefs.as_ref().unwrap().mapv(|x| (100. * x).round() / 100.);

        assert_eq!(arr1(&[cons, lag1, lag2, lag_s]), coefs);

        let y_preds = model.predict(20, None).mapv(|x| (100. * x).round() / 100.);
        y_test = y_test.mapv(|x| (100. * x).round() / 100.);
        assert_eq!(y_test, y_preds);
    }

    #[test]
    fn model_exog() {
        let n_rows = 100;

        let mut x: Array2<f64> = Array::zeros((n_rows, 4));
        x.slice_mut(s![.., 0]).assign(&Array::linspace(-100., -20., n_rows));
        x.slice_mut(s![.., 1]).assign(&Array::logspace(3., 1., 2., n_rows));
        x.slice_mut(s![.., 2]).assign(&Array::logspace(2., -1., 2., n_rows));
        x.slice_mut(s![.., 3]).assign(&Array::geomspace(100., 200., n_rows).unwrap());

        let x_coefs = arr1(&[10.4, 20.6, -10.8, 1.2]);
        let y = x.dot(&x_coefs);

        let y_train = y.slice(s![..80]).to_owned();
        let mut y_test = y.slice(s![80..]).to_owned();
       
        let x_train = x.slice(s![..80, ..]).to_owned();
        let x_test = x.slice(s![80.., ..]).to_owned();
       
        let mut model = Model::moving_average(0);
        model.fit(&y_train, Some(&x_train));

        let coefs = model.coefs.as_ref().unwrap().mapv(|x| (100. * x).round() / 100.);

        assert_eq!(x_coefs, coefs.slice(s![1..]));

        let y_preds = model.predict(20, Some(&x_test)).mapv(|x| (100. * x).round() / 100.);
        y_test = y_test.mapv(|x| (100. * x).round() / 100.);
        assert_eq!(y_test, y_preds);
    }

    #[test]
    #[should_panic(expected = "to be set to 1")]
    fn new_seasonal_s_equal_one() {
        let _model = Model::sarima((1, 2, 3), (4, 5, 6, 1)); 
    }

    #[test]
    fn new_sarima() {
        let model = Model::sarima((1, 2, 3), (4, 5, 6, 7));
        assert_eq!(model.order, Order {p: 1, d: 2, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 4, d: 5, q: 6, s: 7});
    }

    #[test]
    fn new_arima() {
        let model = Model::arima(1, 2, 3);
        assert_eq!(model.order, Order {p: 1, d: 2, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn new_arma() {
        let model = Model::arma(1, 3);
        assert_eq!(model.order, Order {p: 1, d: 0, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn new_ar() {
        let model = Model::autoregressive(1);
        assert_eq!(model.order, Order {p: 1, d: 0, q: 0, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn new_ma() {
        let model = Model::moving_average(3);
        assert_eq!(model.order, Order {p: 0, d: 0, q: 3, s: 1});
        assert_eq!(model.seasonal_order, Order {p: 0, d: 0, q: 0, s: 0});
    }

    #[test]
    fn model_ridership() {
        let y = arr1(&[
            423647.0, 1282779.0, 1361355.0, 1420032.0, 1448343.0, 832757.0, 545656.0, 1575927.0, 1578282.0, 1586936.0, 1603064.0, 1624237.0, 861847.0, 547933.0, 1087994.0, 1646530.0, 1639033.0, 1625828.0, 1493815.0, 846163.0, 550488.0, 1604713.0, 1630335.0, 1598496.0, 1614134.0, 1562363.0, 858914.0, 543253.0, 1540584.0, 1589904.0, 1609900.0, 1669876.0, 1544468.0, 875315.0, 582610.0, 1646532.0, 1651828.0, 1624081.0, 1617997.0, 1564180.0, 876717.0, 584243.0, 1411723.0, 1673587.0, 1615270.0, 1642261.0, 1634458.0, 861011.0, 565643.0, 1240338.0, 1645684.0, 1586345.0, 1592701.0, 1626587.0, 751988.0, 523769.0, 1618585.0, 1609935.0, 1600429.0, 1670329.0, 1673812.0, 965656.0, 564270.0, 1412845.0, 1617091.0, 1645128.0, 1612317.0, 1645040.0, 916723.0, 598309.0, 1568268.0, 1629032.0, 1644700.0, 1557279.0, 1506972.0, 946051.0, 606403.0, 1624169.0, 1652728.0, 1646096.0, 1648556.0, 1636290.0, 881167.0, 519559.0, 1572492.0, 1618818.0, 1609698.0, 1625125.0, 1568647.0, 907173.0, 610433.0, 1634727.0, 1671268.0, 1672982.0, 1579653.0, 1626398.0, 940236.0, 637755.0, 1594708.0, 1638128.0, 1592670.0, 1600486.0, 1392822.0, 908254.0, 528163.0, 1424327.0, 1461693.0, 1507139.0, 1514023.0, 1512741.0, 896740.0, 569781.0, 1570234.0, 1649663.0, 1587241.0, 1652757.0, 1645195.0, 914717.0, 623379.0, 1608814.0, 1700492.0, 1661982.0, 1699840.0, 1667779.0, 944982.0, 624210.0, 1605719.0, 1665983.0, 1661252.0, 1632772.0, 1604363.0, 938175.0, 623266.0, 1544671.0, 1668565.0, 1623862.0, 1635000.0, 1656663.0, 961090.0, 644939.0, 1548440.0, 1656720.0, 1598270.0, 1589150.0, 1586874.0, 782234.0, 598743.0, 584545.0, 1598404.0, 1610382.0, 1568107.0, 1668687.0, 901156.0, 637203.0, 1629146.0, 1541494.0, 1631815.0, 1650925.0, 1654551.0, 997457.0, 680650.0, 1496446.0, 1552051.0, 1510486.0, 1494248.0, 1525695.0, 1016117.0, 660197.0, 1500625.0, 1557150.0, 1552178.0, 1496451.0, 1584079.0, 960183.0, 724860.0, 1568136.0, 1579627.0, 1579268.0, 1553733.0, 1663281.0, 1013073.0, 706369.0, 1586646.0, 1817879.0, 767555.0, 1525074.0, 1601976.0, 946546.0, 695721.0, 1533175.0, 1547135.0, 1575069.0, 1585787.0, 1598341.0, 963025.0, 661964.0, 1503906.0, 1546190.0, 1548192.0, 1568142.0, 1548559.0, 899002.0, 580239.0, 1429003.0, 1569886.0, 1487323.0, 1591522.0, 1568109.0, 965043.0, 668950.0, 1494448.0, 1481119.0, 1486287.0, 1340006.0, 1587888.0, 967792.0, 678516.0, 1493418.0, 1486032.0, 1450530.0, 1448547.0, 1572567.0, 930755.0, 654202.0, 1464101.0, 1464776.0, 1462996.0, 1382289.0, 1527621.0, 866171.0, 675762.0, 1482932.0, 1503237.0, 1463549.0, 1518641.0, 1545290.0, 808837.0, 684729.0, 1491219.0, 1562158.0, 1527106.0, 1530244.0, 1603726.0, 959676.0, 714291.0, 661832.0, 1746122.0, 1692828.0, 1698012.0, 1730163.0, 986759.0, 707799.0, 1736873.0, 1483546.0, 1519672.0, 1617889.0, 1648311.0, 892972.0, 608662.0, 1529366.0, 1665883.0, 1565054.0, 1619206.0, 1593615.0, 948427.0, 551861.0, 1562268.0, 1607378.0, 1667800.0, 1666985.0, 1670026.0, 969472.0, 631889.0, 1709156.0, 1736362.0, 1675558.0, 1552279.0, 1685953.0, 955447.0, 692409.0, 1334023.0, 1700449.0, 1536335.0, 1673065.0, 1625242.0, 789009.0, 618502.0, 1665450.0, 1656443.0, 1677968.0, 1663574.0, 1598544.0, 951492.0, 646652.0, 1634676.0, 1638697.0, 1492245.0, 1565448.0, 1594980.0, 874135.0, 613855.0, 1643028.0, 1623508.0, 1607172.0, 1670898.0, 1688680.0, 963354.0, 666441.0, 1687561.0, 1682926.0, 1674378.0, 1640974.0, 1552106.0, 940977.0, 635049.0, 1339989.0, 1667816.0, 1618624.0, 1598953.0, 1670033.0, 1008038.0, 615882.0, 1600838.0, 1617972.0, 1578084.0, 474703.0, 998426.0, 757962.0, 560171.0, 1594193.0, 1599441.0, 1596247.0, 1585547.0, 1664119.0, 948611.0, 629626.0, 1650328.0, 1662710.0, 1662204.0, 1643449.0, 1642043.0, 935713.0, 614614.0, 1602792.0, 1624849.0, 1590919.0, 1601011.0, 1583784.0, 942189.0, 620795.0, 1588374.0, 1623327.0, 1550605.0, 1574399.0, 1563955.0, 855223.0, 577987.0, 782930.0, 312397.0, 1000031.0, 1127279.0, 1196283.0, 697966.0, 466973.0, 972182.0, 416831.0, 1258483.0, 1361380.0, 1387558.0, 788290.0, 548114.0, 1522736.0, 1558867.0, 1577293.0, 1565930.0, 1598253.0, 832781.0, 547650.0, 1579573.0, 1584579.0, 1563767.0, 1566997.0, 1549261.0, 829812.0, 554972.0, 1041306.0, 1616442.0, 1590911.0, 1613192.0, 1629784.0, 894845.0, 597994.0, 1605529.0, 1577317.0, 1538750.0, 1482883.0, 1539794.0, 888498.0, 581977.0, 1564432.0, 1594271.0, 1609397.0, 1612646.0, 1621538.0, 946488.0, 559710.0, 1577132.0, 1353381.0, 1632301.0, 1623828.0, 1597391.0, 910142.0, 633183.0, 1217537.0, 1582764.0, 1587975.0, 1581077.0, 1608491.0, 922373.0, 620503.0, 1561455.0, 1521122.0, 1527243.0, 1533181.0, 1567078.0, 777614.0, 506605.0, 1220440.0, 1567600.0, 1601341.0, 1603054.0, 1581579.0, 747817.0, 542159.0, 1599706.0, 1633837.0, 1637863.0, 1618205.0, 1597069.0, 957201.0, 579984.0, 1572399.0, 1581091.0, 1584646.0, 1505469.0, 1526262.0, 907328.0, 561842.0, 1440414.0, 1509955.0, 1570117.0, 1536790.0, 1227506.0, 885114.0, 535161.0, 1538023.0, 1555692.0, 1604181.0, 1596005.0, 1628141.0, 957811.0, 535979.0, 1515977.0, 1620153.0, 1639958.0, 1665254.0, 1508731.0, 940202.0, 649776.0, 1512472.0, 1499802.0, 1485717.0, 1491145.0, 1497519.0, 874186.0, 546549.0, 1557348.0, 1642251.0, 1520806.0, 1626973.0, 1641372.0, 785815.0, 566519.0, 1586113.0, 1605025.0, 1636730.0, 1574478.0, 1682207.0, 972358.0, 651828.0, 1592368.0, 1626719.0, 1568944.0, 1623473.0, 1656716.0, 745342.0, 544377.0, 1545853.0, 1608117.0, 1616218.0, 1485967.0, 1574887.0, 937589.0, 623855.0, 1576758.0, 1622513.0, 1618227.0, 1575373.0, 1571895.0, 829563.0, 652742.0, 596086.0, 1548479.0, 1581625.0, 1597432.0, 1676311.0, 1038487.0, 661783.0, 1584925.0, 1490732.0, 1607403.0, 1655623.0, 1642909.0, 972716.0, 675058.0, 1527577.0, 1573549.0, 1593372.0, 1563185.0, 1585587.0, 963347.0, 682062.0, 1556837.0, 1616246.0, 1611377.0, 1603894.0, 1596027.0, 975183.0, 676910.0, 1482392.0, 1527460.0, 1480866.0, 1527728.0, 1584902.0, 1006497.0, 750519.0, 1550475.0, 1577532.0, 1789897.0, 720330.0, 1308516.0, 922711.0, 702237.0, 1504925.0, 1498971.0, 1532721.0, 1563933.0, 1582056.0, 949266.0, 673031.0, 1557792.0, 1567909.0, 1527130.0, 1538931.0, 1573124.0, 974770.0, 641633.0, 1496321.0, 1578240.0, 1563020.0, 1539869.0, 1532990.0, 923464.0, 629953.0, 1458471.0, 1548484.0, 1516407.0, 1577123.0, 1616984.0, 1000269.0, 685764.0, 1511033.0, 1548403.0, 1526040.0, 1534936.0, 1561047.0, 1007716.0, 667858.0, 1488303.0, 1450787.0, 1479665.0, 1519698.0, 1496059.0, 984645.0, 776166.0, 1393301.0, 1505430.0, 1477390.0, 1363690.0, 1473323.0, 934117.0, 668560.0, 1493513.0, 1520528.0, 1512518.0, 1340877.0, 1595849.0, 1011929.0, 734455.0, 582618.0, 1732877.0, 1722590.0, 1718488.0, 1717760.0, 962221.0, 658658.0, 1701659.0, 1691270.0, 1677194.0, 1704876.0, 1719447.0, 970011.0, 642739.0, 1667515.0, 1708990.0, 1682925.0, 1679492.0, 1645476.0, 1005536.0, 632115.0, 1693313.0, 1720538.0, 1730782.0, 1719768.0, 1710343.0, 971035.0, 654059.0, 1682761.0, 1752391.0, 1708715.0, 1756012.0, 1651678.0, 1011012.0, 646285.0, 1728760.0, 1725582.0, 1713298.0, 1727956.0, 1732782.0, 945308.0, 731020.0, 1348875.0, 1721982.0, 1682085.0, 1675890.0, 1702359.0, 988503.0, 646934.0, 1690224.0, 1692769.0, 1648619.0, 1656837.0, 1655907.0, 957215.0, 630941.0, 1669579.0, 1661923.0, 1676738.0, 1635452.0, 1697214.0, 960506.0, 630126.0, 1706107.0, 1607733.0, 1690738.0, 1730706.0, 1642221.0, 959406.0, 639992.0, 1299579.0, 1666942.0, 1671206.0, 1581543.0, 1672961.0, 914224.0, 598343.0, 1638199.0, 1697068.0, 1664428.0, 1605082.0, 1649870.0, 984441.0, 595578.0, 1594729.0, 1571352.0, 1530987.0, 443472.0, 1043665.0, 780419.0, 549342.0, 1565588.0, 1629737.0, 1614060.0, 1613040.0, 1620405.0, 945889.0, 598787.0, 1569738.0, 1632112.0, 1625241.0, 1618350.0, 1621332.0, 979487.0, 651647.0, 1581218.0, 1568856.0, 1511791.0, 1610531.0, 1550838.0, 898744.0, 598533.0, 1294561.0, 941589.0, 334393.0, 1033393.0, 1169471.0, 764693.0, 561278.0, 1243513.0, 1249488.0, 495463.0, 1242621.0, 1377565.0, 826158.0, 554972.0, 1557485.0, 1563218.0, 1604570.0, 1566113.0, 1531499.0, 756529.0, 518385.0, 1559434.0, 1535205.0, 1511682.0, 1537195.0, 1558130.0, 773493.0, 500216.0, 980047.0, 1544983.0, 1484524.0, 1363332.0, 1439854.0, 807231.0, 503829.0, 1486328.0, 1550266.0, 1565193.0, 1594791.0, 1506559.0, 930931.0, 607892.0, 1636104.0, 1577626.0, 1547328.0, 1578861.0, 1527228.0, 833951.0, 548825.0, 1532627.0, 1520638.0, 1272446.0, 1576690.0, 1579851.0, 815768.0, 554215.0, 1157783.0, 1579167.0, 1588040.0, 1599861.0, 1607096.0, 806459.0, 563268.0, 1482337.0, 1470377.0, 1527791.0, 1550713.0, 1609034.0, 901441.0, 529617.0, 1357471.0, 1571010.0, 1452480.0, 1448356.0, 1564757.0, 824616.0, 513758.0, 1490296.0
            ]);
        let n_train: isize = 736;
        let y_train = y.slice(s![..n_train]).to_owned();
        let _y_test = y.slice(s![-n_train..]).to_owned();

        let mut m = Model::sarima((1, 0, 1), (1, 1, 1, 7));
        m.fit(&y_train, None);

        m.predict(10, None);

    }
}