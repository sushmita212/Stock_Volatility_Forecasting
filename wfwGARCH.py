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

    def train_test_split(self):
        cutoff = int(len(self.stock_data) * self.train_size)
        y_test = self.stock_data["returns"].iloc[cutoff:]
        return y_test
    
    def fit_garch(self):
        y_test = self.train_test_split()
        w_fwd_pred =[]
        resids = []
        for i in range(len(y_test)):
            y_train = self.stock_data["returns"].iloc[0:-(len(y_test)-i)]
            model = arch_model(y_train, vol='GARCH', p=1, q=1, rescale=False)
            res= model.fit(disp=0)
            params_dict = res.params.to_dict()
            forecasts = res.forecast(horizon=1)
            pred_vol = (forecasts.variance.values[-1,:][0])**0.5
            w_fwd_pred.append(pred_vol)
            
            # calculate residuals
            actual_return = self.stock_data['returns'].iloc[-(len(y_test)-i)]
            z = actual_return/ pred_vol
            resids.append(z)
        return w_fwd_pred, params_dict, resids

    def plot_results(self,  w_fwd_pred):
        y_test = self.train_test_split()
        w_fwd_vol= pd.Series(w_fwd_pred, index=y_test.index, name='Predicted Volatility')
        test_vol= (1/(4*np.log(2))*(((np.log(self.stock_data['high']/self.stock_data['low'])*100)**2)).rolling(window=1).mean())
        test_vol=np.sqrt(test_vol).iloc[-len(y_test):]
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.index, y_test, color='lightgray', label='Log Returns')
        plt.plot(y_test.index, w_fwd_vol, color='black', label='Predicted Volatility')
        plt.plot(test_vol.index, test_vol, color='red', label='Actual Volatility')
        plt.legend()
        plt.title('GARCH Model Walk Forward Prediction')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.show()


    def evaluate_model(self, w_fwd_pred, print_results=False):
        y_test = self.train_test_split()
        w_fwd_vol = pd.Series(w_fwd_pred, index=y_test.index, name='Predicted Volatility')
        test_vol = (1/(4*np.log(2))*(((np.log(self.stock_data['high']/self.stock_data['low'])*100)**2)).rolling(window=1).mean())
        test_vol = np.sqrt(test_vol).iloc[-len(y_test):]
        
        mae = mean_absolute_error(test_vol, w_fwd_vol)
        mse = mean_squared_error(test_vol, w_fwd_vol)
        rmse = np.sqrt(mse)
        qlike = np.mean(np.log(w_fwd_vol**2)+(test_vol/w_fwd_vol)**2)
        if print_results:
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, QLIKE: {qlike:.4f}")
        return mae, mse, rmse, qlike