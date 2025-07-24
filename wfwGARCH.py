import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera, norm


class clusteringTest:
    def __init__(self, data):
        self.data = data


    def plotACF(self, lags=20):
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_acf(self.data**2, lags=lags, ax=ax)
        plt.title('Autocorrelation Function')
        plt.xlabel('Lags')
        plt.ylabel('ACF')
        plt.show()

    def clusteringScore(self, lags=5, print_score=False):
        acf_values = acf(self.data**2, nlags=lags)
        score = np.sum(acf_values[1:lags+1])
        if print_score:
            print(f'Clustering Score: {score}')
        return round(score, 3)

class GARCHWalkForward:
    def __init__(self, stock_data, ticker, train_size=0.8):
        self.stock_data = stock_data
        self.ticker = ticker
        self.train_size = train_size
        
        
    
    def fit_garch(self):
        cutoff = int(len(self.stock_data) * self.train_size)
        self.y_test = self.stock_data["returns"].iloc[cutoff:]
        
        w_fwd_pred =[]
        resids = []
        params_list = []
        for i in range(len(self.y_test)):
            y_train = self.stock_data["returns"].iloc[0:-(len(self.y_test)-i)]
            model = arch_model(y_train, vol='GARCH', p=1, q=1, rescale=False)
            res= model.fit(disp=0)
            
            # model forecast
            forecasts = res.forecast(horizon=1)
            pred_vol = (forecasts.variance.values[-1,:][0])**0.5
            w_fwd_pred.append(pred_vol)

            # save model parameters
            params_dict = res.params.to_dict()
            params_dict['aic'] = res.aic
            params_dict['bic'] = res.bic
            params_list.append(params_dict)
            
            # calculate residuals
            actual_return = self.stock_data['returns'].iloc[-(len(self.y_test)-i)]
            z = (actual_return-params_dict['mu'])/ pred_vol
            resids.append(z)
        self.residuals = pd.Series(resids)
        self.fit_params = params_list
        self.predictions = w_fwd_pred
        

    def plot_results(self):
        if self.predictions is None or self.y_test is None:
            raise ValueError("Must call fit_garch() before plotting.")
        
        w_fwd_vol= pd.Series(self.predictions, index=self.y_test.index, name='Predicted Volatility')
        test_vol= (1/(4*np.log(2))*(((np.log(self.stock_data['high']/self.stock_data['low'])*100)**2)).rolling(window=1).mean())
        test_vol=np.sqrt(test_vol).iloc[-len(self.y_test):]
        
        # Smoothening for better plot interpretability
        window = 5
        predicted_smooth = w_fwd_vol.rolling(window).mean().dropna()
        realized_smooth = test_vol.rolling(window).mean().dropna()

        # align the index for returns, relaized_smooth and predicted smooth
        common_index = predicted_smooth.index.intersection(realized_smooth.index).intersection(self.y_test.index)
        log_returns = self.y_test.loc[common_index]
        predicted_smooth = predicted_smooth.loc[common_index]
        realized_smooth = realized_smooth.loc[common_index]


        plt.figure(figsize=(10, 6))
        plt.plot(common_index, log_returns, color='lightgray', label='Log Returns')
        plt.plot(common_index, predicted_smooth, color='black',  label='Predicted Volatility')
        plt.plot(common_index, realized_smooth, color='red',alpha=0.7, label='Realized Volatility')
        plt.legend()
        plt.title(f'GARCH Model Walk Forward Prediction for {self.ticker}')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.show()


    def compute_error_metrics(self):
        if self.predictions is None or self.y_test is None:
            raise ValueError("Must call fit_garch() before computing error metrics.")
        
        w_fwd_vol = pd.Series(self.predictions, index=self.y_test.index, name='Predicted Volatility')
        test_vol = (1/(4*np.log(2))*(((np.log(self.stock_data['high']/self.stock_data['low'])*100)**2)).rolling(window=1).mean())
        test_vol = np.sqrt(test_vol).iloc[-len(self.y_test):]
        
        mae = mean_absolute_error(test_vol, w_fwd_vol)
        mse = mean_squared_error(test_vol, w_fwd_vol)
        rmse = np.sqrt(mse)
        qlike = np.mean(np.log(w_fwd_vol**2)+(test_vol/w_fwd_vol)**2)
        
        self.error_metrics = {"MAE":round(mae,3), "MSE": round(mse,3), "RMSE": round(rmse,3), "QLIKE": round(qlike,3)}
        return self.error_metrics


    def residual_diagnostics(self, lags=10, plot=True):
        """
        Perform diagnostic tests and plots on the standardized residuals from the GARCH model.

        Parameters
        ----------
        lags : int, optional
            Number of lags to use for the Ljung-Box test and ACF plot (default is 10).
        plot : bool, optional
            If True, displays diagnostic plots: QQ plot, histogram with normal PDF, and ACF plot (default is True).

        Returns
        -------
        dict
            Dictionary containing p-values for:
                - 'ljung_box_pvalue': Ljung-Box test for autocorrelation
                    Null Hypothesis (H0): Residuals are uncorrelated.
                - 'arch_test_pvalue': ARCH test for conditional heteroskedasticity 
                    Null Hypothesis (H0): No ARCH effects remain.
                - 'jarque_bera_pvalue': Jarque-Bera test for normality 
                    Null Hypothesis (H0): Residuals are normally distributed (zero skewness, kurtosis = 3).

        Raises
        ------
        ValueError
            If residuals are not available (i.e., fit_garch() has not been called).
        """

        if self.residuals is None:
            raise ValueError("Residuals not found. Make sure to call fit_garch() first.")
        
        resids = self.residuals.dropna()

        # Statistical tests
        ljung_p = acorr_ljungbox(resids, lags=[lags], return_df=True).iloc[0]['lb_pvalue']
        arch_p = het_arch(resids)[1]
        jb_p = jarque_bera(resids).pvalue

        if plot:
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            
            sm.qqplot(resids, line='s', ax=axs[0])
            axs[0].set_title('QQ Plot')
            
            # Histogram with normal distribution curve
            axs[1].hist(resids, bins=30, alpha=0.7, edgecolor='black', density=True)
            mu, std = resids.mean(), resids.std()
            xmin, xmax = axs[1].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, loc=mu, scale=std)
            axs[1].plot(x, p, 'r', linewidth=2, label='Normal PDF')
            axs[1].set_title('Residual Histogram')
            axs[1].legend()
            
            sm.graphics.tsa.plot_acf(resids, ax=axs[2])
            axs[2].set_title('ACF of Residuals')
            
            plt.tight_layout()
            plt.show()

        # Store or return test results
        self.resid_test_results = {
            'ljung_box_pvalue': round(ljung_p,3),
            'arch_test_pvalue': round(arch_p,3),
            'jarque_bera_pvalue': round(jb_p,3)
        }

        return self.resid_test_results
