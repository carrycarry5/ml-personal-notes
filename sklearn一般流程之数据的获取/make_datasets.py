# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:46:54 2018

@author: AlanP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# make_blobs()  生成分类、聚类数据
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=200, n_features=2,centers = 4, cluster_std=0.1, center_box=(-5,5) ,random_state=0, shuffle= False)

plt.scatter(X[y == 0, 0], X[y == 0, 1], c = 'red', s = 100, label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c = 'blue', s = 100, label='1')
plt.scatter(X[y == 2, 0], X[y == 2, 1], c = 'green', s = 100, label='2')
plt.scatter(X[y == 3, 0], X[y == 3, 1], c = 'yellow', s = 100, label='3')
plt.legend()
plt.show()



# make_classification()  生成分类数据
#from sklearn.datasets import make_classification
#X, y = make_classification(n_samples=200, n_features=2, n_informative=2,n_redundant=0,
#                           n_repeated=0, n_classes=2, n_clusters_per_class=2,random_state = 1)
#
#plt.scatter(X[y == 0, 0], X[y == 0, 1], c = 'red', s = 100, label='0')
#plt.scatter(X[y == 1, 0], X[y == 1, 1], c = 'blue', s = 100, label='1')
#plt.scatter(X[y == 2, 0], X[y == 2, 1], c = 'green', s = 100, label='2')
#plt.legend()
#plt.show()



# make_circles()   生成二维线性不可分的数据
#from sklearn.datasets import make_circles
#X, y = make_circles(n_samples=200, shuffle=True, noise = 0.05, factor=0.7, random_state=0)
#
#plt.scatter(X[y == 0, 0], X[y == 0, 1], c = 'red', s = 100, label='0')
#plt.scatter(X[y == 1, 0], X[y == 1, 1], c = 'blue', s = 100, label='1')
#plt.scatter(X[y == 2, 0], X[y == 2, 1], c = 'green', s = 100, label='2')
#plt.legend()
#plt.show()


# make_regression  生成回归数据
#from sklearn.datasets import make_regression
#X, y, coef = make_regression(n_samples=200, n_features=1, n_informative=1, n_targets=1,
#                       bias = 0, effective_rank=None, noise = 20,
#                        tail_strength=0,random_state=0, coef = True)
#plt.scatter(X,y)







