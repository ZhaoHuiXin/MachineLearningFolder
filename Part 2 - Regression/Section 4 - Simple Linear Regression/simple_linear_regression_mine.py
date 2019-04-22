# -*- coding: utf-8 -*-
# Simple Linear Regression

# 1.Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2.Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # matrix
y = dataset.iloc[:, 1].values # vector

# omit 3、4
# 5.Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# 接下来将用训练集拟合回归器，用回归器预测测试集

# Feature Scaling， python很多工具里包含了对数据进行特征缩放的这一步，
# 简单线性回归工具里包含了对数据拟合这一步，所以这里没必要处理；
# 而有一些工具包是不包含特征缩放这一步，必须手动进行特征缩放
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


## Fitting Simple Linear Regression to the Training Set
# 用训练集拟合线性回归器
# 1. import class
from sklearn.linear_model import LinearRegression
# 2. create an regressor
regressor = LinearRegression() # >> "Machine"
# 3. fit the regressor by train set
regressor.fit(X_train, y_train) # >> "Learning"


## Predicting the Test set results
# y_pred为包含预测结果的向量
y_pred = regressor.predict(X_test)


## Visualising the "Training" set results, use plt
# draw training set point
plt.scatter(X_train, y_train, color='red')
# draw the regressoin line
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

## Visualising the "Training" set results, use plt
# draw test set point
plt.scatter(X_test, y_test, color='green')
# draw the regressoin line
plt.plot(X_train, regressor.predict(X_train), color='blue')
# the same as above line, because regressor is certain
# plt.plot(X_test, regressor.predict(X_test), color='purple')
plt.title('Salary VS Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
mysalary = regressor.predict([[5]])




