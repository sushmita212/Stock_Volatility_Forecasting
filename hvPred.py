import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class HVmodel:
    def __init__(self, stock_data, train_size=0.8):
        self.stock_data = stock_data
        self.train_size = train_size

    def train_test_split(self):
        cutoff = int(len(self.stock_data) * self.train_size)
        y_test = self.stock_data["returns"].iloc[cutoff:]
        return y_test
    
    def HVpred(self):
        y_test = self.train_test_split()
        hv=self.stock_data['returns'].rolling(window=21).std().shift(1).dropna()
        hv=hv.iloc[-len(y_test):]
    
        return hv

    def plot_results(self,  hv):
        y_test = self.train_test_split()
        h_vol= pd.Series(hv, index=y_test.index, name='Predicted Volatility')
        test_vol= (1/(4*np.log(2))*(((np.log(self.stock_data['high']/self.stock_data['low'])*100)**2)).rolling(window=1).mean())
        test_vol=np.sqrt(test_vol).iloc[-len(y_test):]
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.index, y_test, color='lightgray', label='Log Returns')
        plt.plot(y_test.index, h_vol, color='black', label='Predicted Volatility')
        plt.plot(test_vol.index, test_vol, color='red', label='Actual Volatility')
        plt.legend()
        plt.title('GARCH Model Walk Forward Prediction')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.show()


    def evaluate_model(self, hv, print_results=False    ):
        y_test = self.train_test_split()
        h_vol = pd.Series(hv, index=y_test.index, name='Predicted Volatility')
        test_vol = (1/(4*np.log(2))*(((np.log(self.stock_data['high']/self.stock_data['low'])*100)**2)).rolling(window=1).mean())
        test_vol = np.sqrt(test_vol).iloc[-len(y_test):]
        
        mae = mean_absolute_error(test_vol, h_vol)
        mse = mean_squared_error(test_vol, h_vol)
        rmse = np.sqrt(mse)
        qlike = np.mean(np.log(h_vol**2)+(test_vol/h_vol)**2)

        if print_results:
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, QLIKE: {qlike:.4f}")
        return mae, mse, rmse, qlike
    
