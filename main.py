import numpy as np
import pandas as pd
from pandas_datareader.data import DataReader
from datetime import datetime

etfs = ['VTI', 'EFA', 'EEM', 'TLT', 'TIP', 'VNQ']
train_start = datetime(2005,1,1)
train_end   = datetime(2012,12,31)
test_start  = datetime(2013,1,1)
test_end = datetime(2014,12,31)
train = DataReader(etfs, 'yahoo', start=train_start, end=train_end)['Adj Close']
test  = DataReader(etfs, 'yahoo', start=test_start, end=test_end)['Adj Close']


from preprocessor import ETFScaler
scaler = ETFScaler(train.values, 300)
print('okay')
v = scaler.transfer(train.values[0:1], np.array(100))
print(v)