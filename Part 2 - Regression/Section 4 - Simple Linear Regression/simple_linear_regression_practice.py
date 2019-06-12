#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:01:51 2019

@author: zhx
"""
# import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data read
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:1].values
y = dataset.iloc[:,-1].values

# distribute train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit linear regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predict X_train X_test
y_pred = regressor.predict(X_test)

# plot
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='green')
plt.title('SimpleLinearRegression')
plt.xlabel('age')
plt.ylabel('salary')
plt.show()