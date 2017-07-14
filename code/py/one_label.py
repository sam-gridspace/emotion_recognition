
# coding: utf-8

# In[ ]:

import os
import sys
import csv
import wave
import copy
import math

import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import StratifiedKFold, KFold, train_test_split
from sklearn.svm import OneClassSVM, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import Merge
from keras.layers import LSTM, Input
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop

sys.path.append("../")
from utilities.utils import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#get_ipython().magic(u'matplotlib inline')

#from IPython.display import clear_output


# In[ ]:

batch_size = 64
nb_feat = 34
nb_class = 4
nb_epoch = 80

optimizer = 'Adadelta'


# In[ ]:

params = Constants()
print(params)


# # Calculating features

# In[ ]:

data = read_iemocap_data(params=params)


# In[ ]:

#get_features(data, params)


# # Model definition

# In[ ]:

def build_simple_lstm(nb_feat, nb_class, optimizer='Adadelta'):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(32, nb_feat)))
    model.add(Activation('tanh'))
    model.add(LSTM(256, return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dense(512))
    model.add(Activation('tanh'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# In[ ]:

def build_blstm(nb_feat, nb_class, optimizer='Adadelta'):
    net_input = Input(shape=(78, nb_feat))
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
                          return_sequences=False, 
                          activation="tanh"
                         )(blstm_output1)
    backward_lstm2 = LSTM(output_dim=64, 
                          return_sequences=False, 
                          activation="tanh", 
                          go_backwards=True
                         )(blstm_output1)
    blstm_output2  = Merge(mode='concat')([forward_lstm2, backward_lstm2])
    hidden = Dense(512, activation='tanh')(blstm_output2)
    output = Dense(nb_class, activation='softmax')(hidden)
    model  = Model(net_input, output)
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    
    return model




# # Data preparation

# In[ ]:

X, y, valid_idxs = get_sample(ids=None, take_all=True)
y = to_categorical(y, params)
idxs_train, idxs_test = train_test_split(range(X.shape[0]), test_size=0.2)

print X.shape, y.shape, X[0].shape, X[10].shape

nb_feat = 78
nb_classes = y.shape[1]
# In[ ]:

X, _ = pad_sequence_into_array(X, maxlen=32)

print X.shape

# In[ ]:

X_train, X_test = X[idxs_train], X[idxs_test]
y_train, y_test = y[idxs_train], y[idxs_test]


# # Training

# In[ ]:

# # Model building

# In[ ]:

def build_model():
    return build_simple_lstm(34, nb_classes)

model = build_model()
model.summary()


hist = model.fit(X_train, y_train, 
                 batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, 
                 validation_data=(X_test, y_test))

