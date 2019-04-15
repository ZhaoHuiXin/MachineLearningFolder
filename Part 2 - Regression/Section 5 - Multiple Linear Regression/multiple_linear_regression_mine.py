#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:24:49 2019

@author: zhaohuixin
"""

## Multiple Linear Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# 4.Encoding categorical data
# class LabelEncoder 将不同组的名称转义为数字(就是给类别编号)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# base above transform, then make the sorted number to martix
# categorical_features 要处理的分类数据在数据集里的哪一列
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap, remove col 0 of X 
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## not be need there
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

## Fitting Multiple Linear Regression to the Training set
# library same like Simple Linear Regression, 
# difference is class LinearRegression
from sklearn.linear_model import LinearRegression
# create the regressor(创建多元线性回归器)
regressor = LinearRegression()
# fit the regressor(用训练集拟合多元线性回归器)
regressor.fit(X_train, y_train)

## Predicting the Test set results
# 其实我们要用一个创建好并拟合好的对象，来预测测试集的结果的时候，代码都是一样的
# 首先创建一个包含预测结果的向量
y_pred = regressor.predict(X_test)

# 这里用到了所有的自变量，使用的是  All-in  的策略
# 后面的课程里将对自变量进行筛选，介绍 Backward Elimination 反向淘汰

## Building the optimal model using Backward Elimination
# P-value 越高，代表统计显著性越低
# 1. import library
import statsmodels.formula.api as sm
# 给训练集自变量加上一列值都是1的自变量，见Hands-on @1
# arr 矩阵，会在arr后面加上新的矩阵
# values 我们要加的那个矩阵
# axis=0 add rows to arr, axis=1 add new cols to arr
# 2. transform X_train set
X_train = np.append(arr = np.ones((40, 1)), values=X_train, axis=1)
# optimal，包含最佳的自变量选择, why list col here? 方便后面直接对代码进行改动
X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]
# 用 X_opt 拟合多维线性回归器，这个回归器不是上面的 regressor
# 而是通过新的标准库创建的
# >>反向淘汰 step2 finished, PL is 0.05 here
# 创建新的多维线性回归器并拟合 edog 对应因变量， exog 对应自变量
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
# >>反向淘汰 step3 finshed
regressor_OLS.summary()

# read OLS Regression Results and remove max P-value col x2
X_opt = X_train[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
# read OLS Regression Results and remove max P-value col x1
X_opt = X_train[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
# read OLS Regression Results and remove max P-value col x2
X_opt = X_train[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
# read OLS Regression Results and remove max P-value col x2
X_opt = X_train[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
# 最后行政支出列的P-value是0.7，也不是非常大，但这里的门槛是0.05，还是要舍弃的
# 之后会讲述如何用其他的方法，判断线性模型的性能如何

