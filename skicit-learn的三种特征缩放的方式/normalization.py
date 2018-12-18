# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 09:55:17 2018

@author: AlanP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# impoet the dataset from sklearn
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=300, n_features=4, scale=None,random_state = 0)

# feature scale
# 1.标准的特征缩放
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)


# 2.设置最大最小值的特征缩放
from sklearn import preprocessing
X = preprocessing.minmax_scale(X, feature_range=(0,2))

## 3.普通特征缩放
#from sklearn import preprocessing
#X = preprocessing.scale(X)


# split the train_set and the test_set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

# create a classifier
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# create a confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)   # 96.7%

# get the score of the model
score = classifier.score(X_test, y_test) # 96.7%

# 若不用特征缩放，其准确率为：71.6%

