# Stock volatility prediction
The stock prices of financial assets vary over time. The degree of variation of the stock price is expressed by volatility. Mathematically, the volatility is equivalent to the standard deviation of the stock price over time. Volatily forecasting can be advantageous in trading as it can be used for risk assessment, option pricing, strategy selection etc.
<br>
The goal of this project is to learn and implement time series analysis to forecast volatility of log returns using historical open, high, low, close, volume (OHLCV) data. We use historical daily stock data from Alpha Vantage API.
<br>
Volatility is usually calculated on returns and not on raw stock prices, because returns capture relative change while prices are absolute. For e.g. the same price change of $10 in a $50 stock and a $500 stock means a price change of 20% and 2% respectivley. Thus we need a scale independent measure of volatility which is obtained by volatility on returns. In this project we aim to forecast the volatility of the log return, $r$, of the closing price of of stock given by 
<br>
$$r_t=ln\left(\frac{P_t}{P_{t-1}}\right)$$

## Stock data and volatility clustering
We use stock data from multiple sectors including Finance, ETFs, Healthcare, Technology, and Commodities to enable a more generalized and robust analysis. 

Financial time series data, such as stock returns, often exhibit **volatility clustering** — periods of high volatility tend to be followed by high volatility, and periods of calm tend to persist. For each stock, we compute a clustering score to quantify the degree of volatility clustering. This is defined as:
$\text{Clustering Score} = \sum_{l=1}^5 ACF(r_t^2, lag=l)$,
where ACF is the autocorrelation function. This captures the persistence of volatility, which is characteristic of volatility clustering.

Volatility clustering violates the assumptions of constant variance in simpler models. The GARCH(1,1) model addresses this by modeling the **conditional variance** as a function of past squared returns (recent shocks) and past variances. This makes it well-suited for capturing volatility dynamics and forecasting risk in financial markets.


## GARCH(1,1) model
The GARCH(1,1) models the time-varying volatility (variance) of returns. The returns are of the form
$$r_t=\mu+\epsilon_t$$
where 
- $r_t$ is the return at time $t$, 
- $\mu$ is the modeled mean return, and 
- $\epsilon_t \sim N(0,\sigma_t^2)$ is the residual (shock). 

The conditional variance equation is
$$\sigma_t^2=\omega+\alpha\epsilon_{t-1}^2+\beta\sigma_{t-1}^2$$
where
- $\sigma_t^2$ is the conditional variance at time $t$,
- $\omega>=0$ is the long-run average variace,
- $\alpha>=0$ is the ARCH parameter (reaction to recent shocks),
- $\beta>=0$ is the GARCH parameter (persistence of past variance).

### Walk-forward implementation
To evaluate the performance of the GARCH(1,1) model under realistic conditions we implement a walk-forward validation framework. In this approach:
- The model is retrained at each step using a moving window of historical data. We walk-forward by one time step at each iteration.
- At each iteration, the model forecasts the next-period volatility based only on data available up to that point.

This method mimics how GARCH models would be used in live financial settings, where future data is unknown. This revents data leakage from future observations and provides a more robust estimate of out-of-sample predictive performance. We store the predicted volatility ($\sigma_t$) and the GARCH(1,1) fit parameters $(\mu, \omega, \alpha, \beta)$ at each iteration to evaluate how the model performs over time.

### Model diagnostics
To ensure that the model adequately captures time-varying volatility we analyze the standardized residuals and evolution of model parameters. 
The standardized residuals are calculated as $z_t=(r_t-\mu)/\sigma_t$ at each iteration of the walk-forward loop. If the GARCH model is specified correctly then $z_t$ should have zero mean, constant variance (homoskedastic), be approximately IID (independent and identically distributed). Standardized resoduals often follow a standard normal or standardized t-distribution. 


To verify that this holds for our fitted GARCH model we perform
- **Statistical tests**
(1) Ljung-Box Test: Tests for autocorrelation in residuals.

(2) ARCH Test: Checks for remaining heteroskedasticity.

(3) Jarque-Bera Test: Evaluates normality of residuals.

- **Visual diagnostics**
(1) Histogram of standardized residuals

(2) Q-Q Plot to assess normality

(3) ACF Plot for residuals and squared residuals

(4) Line Plot of $\alpha+\beta$ over time


### Error metrics

## Comparison with the simpler historical volatitlity (HV) prediction 








