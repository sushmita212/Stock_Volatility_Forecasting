import os
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

class stock_data:
    def __init__(self, ticker):
        self.ticker = ticker
        
    
    def get_daily(self,output_size="compact"):
        
        api_key = os.getenv("AV_API_KEY")
        try:
            url = (
                "https://www.alphavantage.co/query?"
                "function=TIME_SERIES_DAILY&"
                f"symbol={self.ticker}&"
                f"outputsize={output_size}&"
                "datatype=json&"
                f"apikey={api_key}"
            )
            response = requests.get(url)
            response_data = response.json()

            # Read data into DataFrame (8.1.12 & 8.1.13)
            stock_data = response_data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(stock_data, orient="index", dtype=float)

            # Convert index to `DatetimeIndex` named "date" (8.1.14)
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
            df.index.name = "date"

            # Remove numbering from columns (8.1.15)
            df.columns = [col.split('. ')[1] for col in df.columns]
            self.records=df
            return df
        except:
            print(response_data)

        # Return DataFrame

    
    def insert_table(self, con, if_exists):
        table_name=self.ticker
        try:
            self.records.to_sql(table_name, con, if_exists=if_exists)
            print(f"Data for {self.ticker} inserted into database")
        except Exception as e:
            print(str(e))
        
    def load_data(self, con):
        table_name = self.ticker
        try:
            sql = f"SELECT * FROM {table_name}"
            df=pd.read_sql(sql, con, parse_dates="data", index_col="date")
            df.index = pd.to_datetime(df.index, format = "ISO8601")
            df.sort_index(inplace=True)
            df["returns"] = np.log(df["close"]/df["close"].shift(1))*100
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(str(e))
        
        
        
        
        
        