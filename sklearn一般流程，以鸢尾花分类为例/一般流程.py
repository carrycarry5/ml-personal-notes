# -*- coding: utf-8 -*-
"""
场景：用鸢尾花数据集为例展示sklearn的一般流程，包括：
1. 数据的获取
2. 数据预处理
3. 模型的训练
4. 模型的评估
5. 模型的优化
6. 模型持久化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1.get the dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2.data preprocession
# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# 3.training the model
from sklearn.svm import SVC
classifier = SVC(kernel = "linear", probability = True)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# 4. 模型的评估
# 查看参数
params = classifier.get_params()

# 查看每种类别的概率
y_pred_proba = classifier.predict_proba(X_test)

# 查看模型评分
score = classifier.score(X_test, y_test)   # 96%

# 分类模型的评分报告
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)

# 交叉验证评分
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(classifier ,X, y, cv = 5)


# 5.模型的优化
from sklearn.model_selection import GridSearchCV
# 估计器
svc = SVC()
# 超参数空间
param_grid = [{'C':[0.1, 1, 10, 100, 1000], 'kernel':['linear']},
               {'C':[0.1, 1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.001, 0.01]}]
# 一共15种情况

# 评分函数
scoring = 'accuracy'

# 指定采样方法, clf即最佳模型
clf = GridSearchCV(svc, param_grid, scoring= scoring, cv = 10) # 返回最佳模型
clf.fit(X_train, y_train)
y_pred_best = clf.predict(X_test) 

score_best = clf.score(X_test, y_pred_best)  # 100%
params_best = clf.get_params()   # 最优模型的参数

# 6. 模型持久化
# pickle
import pickle
with open('clf.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('clf.pkl','rb') as f:
    clf2 = pickle.load(f)

y2_pred = clf2.predict(X_test)

# joblib
from sklearn.externals import joblib
joblib.dump(clf, 'clf_joblib.pkl')
clf3 = joblib.load('clf_joblib.pkl')
y3_pred = clf3.predict(X_test)

