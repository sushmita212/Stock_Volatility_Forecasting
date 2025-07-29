# Stock volatility prediction
The stock prices of financial assets vary over time. The degree of variation of the stock price is expressed by volatility. Mathematically, the volatility is equivalent to the standard deviation of the stock price over time. Volatily forecasting can be advantageous in trading as it can be used for risk assessment, option pricing, strategy selection etc.
<br>
The goal of this project is to learn and implement time series analysis to forecast volatility of log returns using historical open, high, low, close, volume (OHLCV) data with a GARCH(1,1) model. We use historical daily stock data from Alpha Vantage API.
<br>
Stock returns often exhibit volatility clustering — periods of high volatility followed by high volatility. The GARCH(1,1) model captures this by modeling the conditional variance of returns based on past shocks and past volatility.
<br>


## Methodology
- Returns are calculated as log differences of closing prices: $$r_t=ln\left(\frac{P_t}{P_{t-1}}\right)$$.
- Stock data from multiple sectors including Finance, ETFs, Healthcare, Technology, and Commodities is used to enable a more generalized and robust analysis. 
- Each stock is assigned a clustering score based on autocorrelation of squared returns: $$\text{Clustering Score} = \sum_{l=1}^5 ACF(r_t^2, lag=l)$$

## Walk-forward validation
We implement a walk-forward approach to simulate real-world forecasting:
- Retrain the GARCH model on a rolling window
- Forecast next-day volatility
- Store predicted volatility and model parameters at each step

## Model diagnostics
We assess model fit using both statistical tests and visual tools:
- Tests: Ljung-Box (autocorrelation), ARCH (heteroskedasticity), Jarque-Bera (normality) for standardized residuals $z_t=(r_t-\mu)/\sigma_t$.
- Visuals: Histogram, Q-Q plot, ACF plot of standardized residuals, time series of $\alpha+\beta$, and time series of predicted and realized (Parkinson estimator) volatility.

## Evaluation
We compare GARCH volatility forecasts with a Parkinson estimator of realized volatility, and benchmark against historical volatility.

