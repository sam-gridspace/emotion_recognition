
# coding: utf-8

# In[1]:

import os
import sys
import csv
import wave
import copy
import math
import json
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import StratifiedKFold, KFold, train_test_split
from sklearn.svm import OneClassSVM, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier

sys.path.append("../")
from utilities.utils import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().magic(u'matplotlib inline')

from IPython.display import clear_output


# In[2]:

params = Constants()
print(params)


# # Feature calculation

# In[3]:

data = read_iemocap_data()


# In[ ]:

get_features(data, params)


# # Data preparation

# In[ ]:

X, y , _ = get_sample(ids)


# In[ ]:

train_idx, val_idx = train_test_split(X.shape[0], test_size=0.2)


# In[ ]:

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]


# In[ ]:

frames_from_utterance = 2


# In[ ]:

X_train_flattened = X_train[0][X_train[0][:,1].argsort()][::-1][:frames_from_utterance]
appended = X_train_flattened.shape[0]
y_train_flattened = np.array([y_train[0]]*appended)
for utt in range(1, len(X_train)):
    appended = X_train_flattened.shape[0]
    X_train_flattened = np.append(X_train_flattened, 
                                  X_train[utt][X_train[utt][:,1].argsort()][::-1]\
                                  [:frames_from_utterance], 
                                  axis=0)
    appended = X_train_flattened.shape[0] - appended
    y_train_flattened = np.append(y_train_flattened,
                                  [y_train[utt]]*appended, 
                                  axis=0)
y_train_binary = copy.deepcopy(y_train_flattened)
y_train_binary = np.argmax(to_categorical(y_train_binary), axis=1)
y_test_binary = np.argmax(to_categorical(y_val), axis=1)


# # MultiClass Probability RFC

# In[18]:

clf = RandomForestClassifier(n_estimators=1000,
                             class_weight="balanced")
clf.fit(X_train_flattened, y_train_binary)

preds = [clf.predict_proba(X_val[0])]
for utt in range(1, len(X_val)):
    preds.append(clf.predict_proba(X_val[utt]))

a = []
for i in preds:
    a.append(Counter(np.argmax(i, axis=1)).most_common(1)[0][0])

WA = weighted_accuracy(y_test_binary, a)
UA = unweighted_accuracy(y_test_binary, a)

