import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

class HVmodel:
    def __init__(self, stock_data, train_size=0.8):
        self.stock_data = stock_data
        self.train_size = train_size

    
    def HVpred(self):
        cutoff = int(len(self.stock_data) * self.train_size)
        self.y_test = self.stock_data["returns"].iloc[cutoff:]
        
        # Shift HV prediction by 1 because for step t+1 we can only use data up to t
        hv=self.stock_data['returns'].rolling(window=21).std().shift(1).dropna()
        hv=hv.iloc[-len(self.y_test):]
        self.hv_pred = hv

    def plot_results(self):
        if self.hv_pred is None or self.y_test is None:
            raise ValueError("Must call HVpred() before plotting.")
        
        h_vol= pd.Series(self.hv_pred, index=self.y_test.index, name='Predicted Volatility')

        test_vol= (1/(4*np.log(2))*(((np.log(self.stock_data['high']/self.stock_data['low'])*100)**2)).rolling(window=1).mean())
        test_vol=np.sqrt(test_vol).iloc[-len(self.y_test):]


        plt.figure(figsize=(10, 6))
        plt.plot(self.y_test.index, self.y_test, color='lightgray', label='Log Returns')
        plt.plot(self.y_test.index, h_vol, color='black', label='Predicted Volatility')
        plt.plot(test_vol.index, test_vol, color='red', label='Realized Volatility')
        plt.legend()
        plt.title('GARCH Model Walk Forward Prediction')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.show()


    def compute_error_metrics(self):
        if self.hv_pred is None or self.y_test is None:
            raise ValueError("Must call HVpred() before computing error metrics.")
        
        h_vol = pd.Series(self.hv_pred, index=self.y_test.index, name='Predicted Volatility')
       
        test_vol = (1/(4*np.log(2))*(((np.log(self.stock_data['high']/self.stock_data['low'])*100)**2)).rolling(window=1).mean())
        test_vol = np.sqrt(test_vol).iloc[-len(self.y_test):]

        mae = mean_absolute_error(test_vol, h_vol)
        mse = mean_squared_error(test_vol, h_vol)
        rmse = np.sqrt(mse)
        qlike = np.mean(np.log(h_vol**2)+(test_vol/h_vol)**2)

        self.error_metrics = {"MAE":round(mae,3), "MSE": round(mse,3), "RMSE": round(rmse,3), "QLIKE": round(qlike,3)}
        return self.error_metrics
    
