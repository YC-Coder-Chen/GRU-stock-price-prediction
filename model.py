#!/usr/bin/env python3

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, rnn, loss as gloss

"Define the attention model"
def attention_function(attention_size):
    model = nn.Sequential()
    model.add(nn.Dense(attention_size, activation='tanh', use_bias=False, flatten=False),
              nn.Dense(1, use_bias=False, flatten=False))
    return model

"Define the function to perfrom attention mechanism"
def attention_forward(attention, cur_features, cur_state):
    """
    cur_features: (batch_size, num_features)
    cur_state: (batch_size, num_hidden)
    
    """
    
    cur_features = cur_features.T # (num_features, batch_size)
    cur_state = cur_state.expand_dims(0) # (1, batch_size, num_hidden)
    cur_states = nd.broadcast_axis(cur_state, axis=0, size=cur_features.shape[0])
    cur_features = cur_features.expand_dims(2)
    features_and_cur_states = nd.concat(cur_features, cur_states, dim=2)
    
    """
    features_and_cur_states: (num_features, batch_size, num_hiddens + 1)
    attention(features_and_cur_states): (num_features, batch_size, 1)
    alpha_prob: (num_features, batch_size, 1)
    
    """
    
    alpha_prob = nd.softmax(attention(features_and_cur_states), axis=0)
    return (alpha_prob * cur_states).sum(axis=0)

"Define the GRU model"
class GRU_Model(nn.Block):
    def __init__(self, num_hiddens, num_layers, attention_size, drop_prob=0.5, **kwargs):
        super(GRU_Model, self).__init__(**kwargs)
        self.attention = attention_function(attention_size)
        self.gru = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        self.dense = nn.Dense(1)

    def forward(self, cur_features, cur_price, state):
        # Apply Attention Mechanism, use the last hidden state of encoder as enc_state
        """
        cur_features: (batch_size, num_features)
        cur_price: (batch_size, 1)
        state: (batch_size, num_hiddens) 
        context: (batch_size, 1) # the weighted avg of cur_price and state[-1]
        input_context: (batch_size, 1 + num_hidden)
        
        """
        context = attention_forward(self.attention, cur_features, state[0][-1])
        input_context = nd.concat(cur_price.expand_dims(1), context, dim=1)
        output, state = self.gru(input_context.reshape(1, cur_price.shape[0], -1), state)
        
        output = nd.concat(output.reshape(output.shape[1], output.shape[2]), cur_price.reshape(-1,1), dim=1)
        output = self.dense(output).reshape(-1, 1)
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.gru.begin_state(*args, **kwargs)
    
"Define the loss function"
def loss_func(net, inputs, features, outputs, loss, ctx):
    batch_size = inputs.shape[0] #in case batch_size is changed
    
    # in each batch, we will reinitialize the state
    state = net.begin_state(batch_size = batch_size, ctx = ctx)
    l = nd.array([0], ctx = ctx)
    n = 0
    for idx in range(inputs.shape[1]):
        cur_features = features.swapaxes(0, 1)[idx] # (batch_size, num_features)
        cur_price =inputs.swapaxes(0, 1)[idx] # (batch_size, num_features)
        output, state = net(cur_features, cur_price, state)
        l = l + (loss(output.reshape(batch_size,), outputs.swapaxes(0, 1)[idx])).sum()
        n = n + batch_size
    return l / n

"Define the loss matrix for the model"
def evaluate_test(data_iter, net, ctx, loss = gloss.L1Loss()):
    
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for inputs, features, outputs in data_iter:
        batch_size = inputs.shape[0] #in case batch_size is changed
        state = net.begin_state(batch_size = batch_size, ctx = ctx)
        inputs = inputs.as_in_context(ctx).astype('float32')
        features = features.as_in_context(ctx).astype('float32')
        outputs = outputs.as_in_context(ctx).astype('float32')
        for idx in range(inputs.shape[1]):
            cur_features = features.swapaxes(0, 1)[idx] # (batch_size, num_features)
            cur_price =inputs.swapaxes(0, 1)[idx] # (batch_size, num_features)
            output, state = net(cur_features, cur_price, state)
            acc_sum = acc_sum + (loss(output.reshape(batch_size,), outputs.swapaxes(0, 1)[idx])).sum()
            n = n + batch_size
    return acc_sum.asscalar() / n

"Define the train function"
def train(net, train_iter, test_iter, lr, batch_size, num_epochs, loss, ctx):
    net.initialize(ctx = ctx, init = init.Xavier(), force_reinit=True)   
    
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    
    for epoch in range(num_epochs):
        l_sum = 0.0
        for inputs, features, outputs in train_iter:
            inputs = inputs.as_in_context(ctx).astype('float32')
            features = features.as_in_context(ctx).astype('float32')
            outputs = outputs.as_in_context(ctx).astype('float32')
            
            with autograd.record():
                l = loss_func(net, inputs, features, outputs, loss, ctx)
            l.backward()
            trainer.step(1) # we already calculate the avg
            l_sum += l.asscalar()
        if (epoch + 1) % 50 == 0:
            test_acc = evaluate_test(test_iter, net, ctx)
            print(f"epoch {epoch + 1}, train loss {l_sum / len(train_iter)}, test loss {test_acc}")
            net.save_parameters(f'./data/params_encoder_{epoch + 1}')