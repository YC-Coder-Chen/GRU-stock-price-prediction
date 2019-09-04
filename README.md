# GRU-stock-price-prediction


![Final Prediction](/prediction.png)
Welcome, this project proposes a new GRU framework for stock price prediction using MXNET and Python 3.6. Previous LSTM/GRU model can only take in previous stock price. But in fact, stock price changes are clearly not determined by previous stock price solely. So we created this new framework incorporated with a mechanism very similar to the attention mechanism to takes in other features (such as previous day's highest price changes/ volumes changes). We also incorporated a mechanism quite similar to the ResNet in computer vision industry into the model. For simplification, this code demo only uses time-series train/test split. After carefully tuning, this framework achieved a slightly better result than simply using the lag of stock price as the prediction. The reason behind this might be there isn't much information behind the static trading data. 

Data
------------

The provided dataset came from Yahoo Finance APPLE trading data. The whole dataset starts from 2015.12.30 and ends at 2018.08.31. We use the head 80% of the data as trainset and the latter part of the dataset as the test set.

Model training
------------

User can modify the [train_model.py](/train_model.py) and run the file to train your model.  
The default optimizer is "Adam", users can also change the optimizer to "SGD" or other optimizers supported by MXNET in the [model.py](/model.py). More specific parameters details are provided in the file. Below is the setting parameters for the trained model.

```
# for the dataset
batch_size = 32
batch_length = 20 # in each sample of each batch, we only predict 20 period stock price 


# for the model
num_hiddens = 10
num_layers = 3
attention_size = 10
drop_prob = 0.5 # prevent overfit

# for the training
lr = 0.001
num_epochs = 750 # should be decided by the CV, but for simplification, we skip it
loss = gloss.L1Loss()
ctx = mx.cpu()
```
