# -*- coding: utf-8 -*-

# multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# encoding categorical data  # 处理分类数据,虚拟编码
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
X[:,3] = label_encoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])   # 表示要处理第3列
X = onehotencoder.fit_transform(X).toarray()
print("---------------")
print(type(onehotencoder.fit_transform(X)))   #  这里可以看出，onehotencoder.fit_transform(X)是一个scipy格式
print(type(X))                                   # X是一个numpy格式，scipy格式转numpy使用toarray()函数

# 采用虚拟编码会出现一个问题——虚拟编码陷阱
# Avoiding the Dummy Variable Trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 我们用到的标准库会直接对数据集进行特征缩放，所以无需手动缩放

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # 创建回归器
regressor.fit(X_train, y_train)  # 用训练集去拟合回归器

# Predicting the Test set results
y_pred = regressor.predict(X_test) #获得预测结果
plt.plot(y_test, c="red")
plt.plot(y_pred)
plt.show()

# Building the optional model using Backward Elimination
# 运用反向淘汰选择模型
# 并不是所有的特征对于整个模型来说都是非常重要的，其中有一些特征对模型的影响非常小，
# 我们应该淘汰掉这些特征。
import statsmodels.formula.api as sm
# 给训练集加上一个全1列
X_train = np.append(arr = np.ones((40,1)), values=X_train, axis=1)

X_opt = X_train[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
# 经检测，发现x2的P>|t|(p_value)最大，所以把这一列去掉
X_opt = X_train[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
# 经检测，发现x1的p>|t|(p_value)最大，所以把这一列去掉
X_opt = X_train[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
# 经检测，发现x2的p>|t|(p_value)最大，所以把这一列去掉
X_opt = X_train[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
# 经检测，发现x2的p>|t|(p_value)最大，所以把这一列去掉
X_opt = X_train[:,[0,3]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
# 最终发现，研发金额对市值的影响最大