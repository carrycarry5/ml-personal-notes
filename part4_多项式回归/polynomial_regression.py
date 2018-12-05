# -*- coding: utf-8 -*-

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # 自变量应该是矩阵
y = dataset.iloc[:, 2].values    # 因变量应该是向量

# Splitting the dataset into the Training set and Test set
# 由于数据集太小，所以不考虑分成训练集和测试集
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
# 同样也不用做特征缩放，因为函数内部已经帮我们做了


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)  # 拟合线性模型

# Fitting Polynomail Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # 产生一个最高维度是4的矩阵
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)  # 拟合多项式线性回归

# visuallising the Linear Regression results
plt.scatter(X, y, c="red")
plt.plot(X, lin_reg.predict(X))
plt.title = "Truth or Bluff (Linear Regression)"
plt.xlabel = "possion level"
plt.ylabel = "salary"
plt.show()

# Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1) # 这样将x分成多段，可以使生成的曲线更平滑
X_grid = X_grid.reshape(len(X_grid), 1)  # 将向量转换成了矩阵
plt.scatter(X, y, c="red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)))
plt.title = "Truth or Bluff (polynimal regression)"
plt.xlabel = "possion level"
plt.ylabel = "salary"
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)  # 33w

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))  # 15.8w


