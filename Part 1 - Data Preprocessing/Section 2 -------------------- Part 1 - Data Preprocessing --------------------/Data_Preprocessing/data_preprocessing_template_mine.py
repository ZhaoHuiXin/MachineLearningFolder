#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:00:27 2019

@author: zhaohuixin
"""
# 1.Importing the libraries
import numpy as np # math methods
import matplotlib.pyplot as plt # draw
import pandas as pd # operate data


# 2.Importing the dataset
dataset = pd.read_csv('Data.csv')

# X包含自变量的矩阵 iloc fetch some row and col
# ,左边代表取得行数，右边代表取得列数（除去最后一列）
X = dataset.iloc[:, :-1].values

# y包含因变量的向量, 取所有行的第三列
y = dataset.iloc[:,3].values

# 3.Taking care of missing data
# scikit-learn: a data dive and analysis library
# Imputer: an class deal missing strategy
# tip: command + i
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
# fit,拟合 1:3 mean 1~2 not include 3
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# 4.Encoding categorical data
# class LabelEncoder 将不同组的名称转义为数字(就是给类别编号)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# base above transform, then make the sorted number to martix
# categorical_features 要处理的分类数据在数据集里的哪一列
onhotencoder = OneHotEncoder(categorical_features = [0])
X = onhotencoder.fit_transform(X).toarray()
# 处理因变量，只转化成数字即可
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# 5.Splitting the datase into the Training set and Test set
from sklearn.model_selection import train_test_split
# 训练集自变量， 测试集自变量， 训练集因变量， 测试集因变量
# test_size 测试集所占比例
# random_state 随机分配数据到测试集和训练集，是整数
# random_state决定随机数生成的方式,random_state都是相同的话会得到完全一样的训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state=0)

# 6.Feature Scaling, 这里对虚拟变量也进行了特征缩放
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# 拟合，用训练集的数据
X_train = sc_X.fit_transform(X_train)
# 上一步sc_X已经拟合好了，此时直接transform即可
X_test = sc_X.transform(X_test)







