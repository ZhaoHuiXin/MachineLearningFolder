#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:37:42 2019

@author: zhx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
line_reg = LinearRegression()
line_reg = line_reg.fit(X, y)
y_pred = line_reg.predict(X)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

line_reg2 = LinearRegression()
line_reg2 = line_reg2.fit(X_poly, y)
y_pred2 = line_reg2.predict(X_poly)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X, y, color='red')
plt.plot(X_grid, line_reg2.predict(poly_reg.fit_transform(X_grid)), color='purple')
plt.show()