
# coding: utf-8

# In[1]:

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
import json

from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import StratifiedKFold, KFold, train_test_split

import keras.backend as K
from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense, Activation, Merge, Dropout
from keras.layers import LSTM, Input, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop

sys.path.append("../")
from utilities.utils import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().magic(u'matplotlib inline')

from IPython.display import clear_output


# In[2]:

batch_size = 64
nb_feat = 34
nb_class = 4
nb_epoch = 2

optimizer = 'Adadelta'


# In[3]:

params = Constants()
print(params)


# In[4]:

params.path_to_data = "/root/shared/Dropbox/study/Skoltech/voice/data/initial/IEMOCAP_full_release/"


# # Calculating features

# In[5]:

data = read_iemocap_data(params=params)


# In[6]:

get_features(data, params)


# # Model definition

# In[7]:

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    shift = 2
    y_pred = y_pred[:, shift:, :]
    input_length -= shift
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# In[8]:

def build_model(nb_feat, nb_class, optimizer='Adadelta'):
    net_input = Input(name="the_input", shape=(78, nb_feat))
    forward_lstm1  = LSTM(output_dim=64, 
                          return_sequences=True, 
                          activation="tanh"
                         )(net_input)
    backward_lstm1 = LSTM(output_dim=64, 
                          return_sequences=True, 
                          activation="tanh",
                          go_backwards=True
                         )(net_input)
    blstm_output1  = Merge(mode='concat')([forward_lstm1, backward_lstm1])

    forward_lstm2  = LSTM(output_dim=64, 
                          return_sequences=True, 
                          activation="tanh"
                         )(blstm_output1)
    backward_lstm2 = LSTM(output_dim=64, 
                          return_sequences=True, 
                          activation="tanh",
                          go_backwards=True
                         )(blstm_output1)
    blstm_output2  = Merge(mode='concat')([forward_lstm2, backward_lstm2])

    hidden = TimeDistributed(Dense(512, activation='tanh'))(blstm_output2)
    output = TimeDistributed(Dense(nb_class + 1, activation='softmax'))(hidden)

    labels = Input(name='the_labels', shape=[1], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([output, labels, input_length, label_length])

    model = Model(input=[net_input, labels, input_length, label_length], output=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer, metrics=[])

    test_func = K.function([net_input], [output])
    
    return model, test_func


# # Model building

# In[9]:

model, test_func = build_model(nb_feat=nb_feat, nb_class=nb_class, optimizer=optimizer)
model.summary()


# # Data preparation

# In[10]:

X, y, valid_idxs = get_sample(ids=None, take_all=True)
y = np.argmax(to_categorical(y, params), axis=1)
y = np.reshape(y, (y.shape[0], 1))


# In[11]:

X, X_mask = pad_sequence_into_array(X, maxlen=78)
y, y_mask = pad_sequence_into_array(y, maxlen=1)


# In[12]:

index_to_retain = np.sum(X_mask, axis=1, dtype=np.int32) > 5


# In[13]:

X, X_mask = X[index_to_retain], X_mask[index_to_retain]
y, y_mask = y[index_to_retain], y_mask[index_to_retain]


# In[14]:

idxs_train, idxs_test = train_test_split(range(X.shape[0]))
X_train, X_test = X[idxs_train], X[idxs_test]
X_train_mask, X_test_mask = X_mask[idxs_train], X_mask[idxs_test]
y_train, y_test = y[idxs_train], y[idxs_test]
y_train_mask, y_test_mask = y_mask[idxs_train], y_mask[idxs_test]


# # Training

# In[15]:

sess = tf.Session()


# In[16]:

class_weights = np.unique(y, return_counts=True)[1]*1.
class_weights = np.sum(class_weights) / class_weights

sample_weight = np.zeros(y_train.shape[0])
for num, i in enumerate(y_train):
    sample_weight[num] = class_weights[i[0]]


# In[17]:

ua_train = np.zeros(nb_epoch)
ua_test = np.zeros(nb_epoch)
wa_train = np.zeros(nb_epoch)
wa_test = np.zeros(nb_epoch)
loss_train = np.zeros(nb_epoch)
loss_test = np.zeros(nb_epoch)

for epoch in range(nb_epoch):
    epoch_time0 = time.time()
    
    total_ctcloss = 0.0
    batches = range(0, X_train.shape[0], batch_size)
    shuffle = np.random.choice(batches, size=len(batches), replace=False)
    for num, i in enumerate(shuffle):
        inputs_train = {'the_input': X_train[i:i+batch_size],
                        'the_labels': y_train[i:i+batch_size],
                        'input_length': np.sum(X_train_mask[i:i+batch_size], axis=1, dtype=np.int32),
                        'label_length': np.squeeze(y_train_mask[i:i+batch_size]),
                       }
        outputs_train = {'ctc': np.zeros([inputs_train["the_labels"].shape[0]])}

        ctcloss = model.train_on_batch(x=inputs_train, y=outputs_train, 
                                       sample_weight=sample_weight[i:i+batch_size])

        total_ctcloss += ctcloss * inputs_train["the_input"].shape[0] * 1.
    loss_train[epoch] = total_ctcloss / X_train.shape[0]

    inputs_train = {'the_input': X_train,
                    'the_labels': y_train,
                    'input_length': np.sum(X_train_mask, axis=1, dtype=np.int32),
                    'label_length': np.squeeze(y_train_mask),
                   }
    outputs_train = {'ctc': np.zeros([y_train.shape[0]])}
    preds = test_func([inputs_train["the_input"]])[0]
    decode_function = K.ctc_decode(preds[:,2:,:], inputs_train["input_length"]-2, greedy=False, top_paths=1)
    labellings = decode_function[0][0].eval(session=sess)
    if labellings.shape[1] == 0:
        ua_train[epoch] = 0.0
        wa_train[epoch] = 0.0
    else:
        ua_train[epoch] = unweighted_accuracy(y_train.ravel(), labellings.T[0].ravel())
        wa_train[epoch] = weighted_accuracy(y_train.ravel(), labellings.T[0].ravel())


    inputs_test = {'the_input': X_test,
                   'the_labels': y_test,
                   'input_length': np.sum(X_test_mask, axis=1, dtype=np.int32),
                   'label_length': np.squeeze(y_test_mask),
                  }
    outputs_test = {'ctc': np.zeros([y_test.shape[0]])}
    preds = test_func([inputs_test["the_input"]])[0]
    decode_function = K.ctc_decode(preds[:,2:,:], inputs_test["input_length"]-2, greedy=False, top_paths=1)
    labellings = decode_function[0][0].eval(session=sess)
    if labellings.shape[1] == 0:
        ua_test[epoch] = 0.0
        wa_test[epoch] = 0.0
    else:
        ua_test[epoch] = unweighted_accuracy(y_test.ravel(), labellings.T[0].ravel())
        wa_test[epoch] = weighted_accuracy(y_test.ravel(), labellings.T[0].ravel())
    loss_test[epoch] = np.mean(model.predict(inputs_test))

    epoch_time1 = time.time()


    print('epoch = %d, WA Tr = %0.2f, UA Tr = %0.2f, WA Te = %0.2f, UA Te = %0.2f, CTC Tr = %0.2f, CTC Te = %0.2f, time = %0.2fmins' % (epoch + 1, 
                     wa_train[epoch], ua_train[epoch], 
                     wa_test[epoch], ua_test[epoch], 
                     loss_train[epoch], loss_test[epoch],
                     (epoch_time1-epoch_time0)/60))

