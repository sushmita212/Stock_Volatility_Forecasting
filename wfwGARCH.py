import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf


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
    def __init__(self, stock_data, train_size=0.8):
        self.stock_data = stock_data
        self.train_size = train_size
        self.residuals = None
        self.fit_params = None
        self.predictions = None
        self.y_test = None
        self.error_metrics = None
        
    
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
            params_list.append(params_dict)
            
            # calculate residuals
            actual_return = self.stock_data['returns'].iloc[-(len(self.y_test)-i)]
            z = actual_return/ pred_vol
            resids.append(z)
        self.residuals = resids
        self.fit_params = params_list
        self.predictions = w_fwd_pred
        

    def plot_results(self):
        if self.predictions is None or self.y_test is None:
            raise ValueError("Must call fit_garch() before plotting.")
        
        w_fwd_vol= pd.Series(self.predictions, index=self.y_test.index, name='Predicted Volatility')
        test_vol= (1/(4*np.log(2))*(((np.log(self.stock_data['high']/self.stock_data['low'])*100)**2)).rolling(window=1).mean())
        test_vol=np.sqrt(test_vol).iloc[-len(self.y_test):]
        plt.figure(figsize=(10, 6))
        plt.plot(self.y_test.index, self.y_test, color='lightgray', label='Log Returns')
        plt.plot(self.y_test.index, w_fwd_vol, color='black', label='Predicted Volatility')
        plt.plot(test_vol.index, test_vol, color='red', label='Actual Volatility')
        plt.legend()
        plt.title('GARCH Model Walk Forward Prediction')
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
        
        self.error_metrics = {"MAE":mae, "MSE": mse, "RMSE": rmse, "QLIKE": qlike}
        
        