# -*- coding: utf-8 -*-

# simple Linear Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # 创建回归器
regressor.fit(X_train, y_train)  # 

# predicting the Test set results
y_pred = regressor.predict(X_test)    # pred:predict(预测),给测试集的数据做预测

# Visualising the Training set results
plt.scatter(X_train, y_train, c='red')   # 散点图,训练集的
plt.plot(X_train,  regressor.predict(X_train),c='blue')
plt.title = "salary vs experience（training set）"
plt.xlabel = "years of experience"
plt.ylabel = "salary"
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


