#!/usr/bin/env python3

import datetime
import numpy as np
import pickle
import pandas_datareader.data as web
import mxnet as mx
from mxnet import nd
from mxnet.gluon import data as gdata, loss as gloss
from model import GRU_Model, train
import pylab
import matplotlib.pyplot as plt

# fetch the data
start = datetime.datetime(2005,12,30)
end = datetime.datetime(2018,8,31)
data = web.get_data_yahoo(['AAPL','^NDX'], start,end)
data.columns = ['_'.join(col).strip() for col in data.columns.values]
data['Outcome'] = np.nan
data['Outcome'].iloc[0:-1] = data['Close_AAPL'].iloc[1:].values # create the target

# we use information other than stock price as features input
data['Close_pct'] = data['Close_AAPL'].pct_change() # convert actual values into pct
feature_col = list(set(data.columns) - set(['Close_AAPL','Outcome','Close_pct']))
data[feature_col] = data[feature_col].pct_change()

"""
features contains: ['Adj Close_^NDX',
                    'Adj Close_AAPL',
                    'Open_^NDX',
                    'Open_AAPL',
                    'High_AAPL',
                    'Volume_AAPL',
                    'Low_^NDX',
                    'Low_AAPL',
                    'Volume_^NDX',
                    'High_^NDX',
                    'Close_^NDX']
"""
data = data.dropna().copy()
feature_col = feature_col + ['Close_pct']  # add the previous day's stock price changes into features

# for simplification, we only split the data into train and test
train_data = data.iloc[0:int(len(data)*0.8)]
test_data = data.iloc[int(len(data)*0.8):]

# create the gdata object
trainset = gdata.ArrayDataset(train_data['Close_AAPL'].values[0:2520].reshape(-1, 20).tolist(),
                              train_data[feature_col].values[0:2520].reshape(-1, 20, 12).tolist(), 
                              train_data['Outcome'].values[0:2520].reshape(-1, 20).tolist())
testset = gdata.ArrayDataset(test_data['Close_AAPL'].values[0:620].reshape(-1, 20).tolist(),
                             test_data[feature_col].values[0:620].reshape(-1, 20, 12).tolist(), 
                             test_data['Outcome'].values[0:620].reshape(-1, 20).tolist())

# save the train and test set
batch_size = 32
train_iter = gdata.DataLoader(trainset, batch_size, shuffle=True)
test_iter = gdata.DataLoader(testset, batch_size, shuffle=False)

with open(f"./data/train_iter.pkl", "wb") as fp:
    pickle.dump(train_iter, fp) 
with open(f"./data/test_iter.pkl", "wb") as fp:
    pickle.dump(test_iter, fp) 

# show the trainset batch shape
for x, y, z in train_iter:
    break
print(x.shape) # model input
print(y.shape) # attention input
print(z.shape) # output

# define the model
net = GRU_Model(num_hiddens = 10, 
                 num_layers = 3, 
                 attention_size = 10,
                 drop_prob=0.5)
lr = 0.001
batch_size = 32
num_epochs = 750 # should be decided by the CV, but for simplification, we skip it
loss = gloss.L1Loss()
ctx = mx.cpu()

train(net, train_iter, test_iter, lr, batch_size, num_epochs, loss, ctx)

# plot the result

train_data = train_data.reset_index()
prediction = []
state = net.begin_state(batch_size = 1, ctx = ctx)
for idx in range(len(train_data)):
    cur_price = nd.array([train_data.loc[idx,'Close_AAPL']]).reshape(1,)
    cur_features = nd.array([train_data.loc[0,feature_col]])
    output, state = net(cur_features, cur_price, state)
    prediction.append(output.reshape(1,).asscalar())
train_data['Prediction'] = prediction


test_data = test_data.reset_index()
prediction = []
for idx in range(len(test_data)):
    cur_price = nd.array([test_data.loc[idx,'Close_AAPL']]).reshape(1,)
    cur_features = nd.array([test_data.loc[0,feature_col]])
    output, state = net(cur_features, cur_price, state)
    prediction.append(output.reshape(1,).asscalar())
test_data['Prediction'] = prediction

print('The loss for the prediction by GRU',str((test_data['Outcome'] - test_data['Prediction']).abs().sum()/len(test_data)))
print('The loss for the prediction if we only use lag',str((test_data['Outcome'] - test_data['Close_AAPL']).abs().sum()/len(test_data)))

fig = plt.figure(figsize=(15,4))
pylab.plot(test_data['Outcome'], '-g', label='Actual')
pylab.plot(test_data['Prediction'], '-b', label='Prediction')
pylab.plot(test_data['Close_AAPL'], '-r', label='Lag')
pylab.legend(loc='lower right')
pylab.xlabel('Time', fontsize=10)
pylab.ylabel('Stock Price', fontsize=10)
fig.suptitle('APPL Stock Price- testset', fontsize=20)

