# -*- coding: utf-8 -*-

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('../Data.csv')
x1 = dataset.iloc[:, :-1]    # dataframe格式
x = dataset.iloc[:, :-1].values   # numpy格式  自变量应该是个矩阵
y = dataset.iloc[:,-1:].values    # 因变量应该是个向量
print("hello spyter")

# taking care of the missing data
from sklearn.preprocessing import Imputer  #该类专门用于处理缺失对象
imputer = Imputer(missing_values= 'NaN',strategy='mean',axis=0) #axis=0代表对列操作
imputer.fit(x[:,1:3])  # 用x去拟合
x[:,1:3] = imputer.transform(x[:,1:3])  


# encoding categorical data  # 处理分类数据
from sklearn.preprocessing import LabelEncoder,OneHotEncoder # 该类用于将label标准化
label_encoder_x = LabelEncoder()  # 转换'国家'
x[:,0] = label_encoder_x.fit_transform(x[:,0]) # 拟合、转换
label_encoder_y = LabelEncoder() # 转换'是否购买'
y = label_encoder_y.fit_transform(y[:,0])

# 虚拟编码
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

# sptting the dataset into the Training set and Test set  划分训练集和测试集
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train[:,-2:] = sc_x.fit_transform(x_train[:,-2:])
x_test[:,-2:] = sc_x.transform(x_test[:,-2:]) # 之前sc_x已经拟合过了，所以不用再拟合了
