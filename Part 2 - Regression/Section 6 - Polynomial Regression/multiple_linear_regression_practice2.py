#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:22:19 2019

@author: zhx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.preprocessing import PolynomialFeatures
X_grid = PolynomialFeatures(degree=4).fit_transform(X)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_grid, y)

y_pred = regressor.predict(X_grid)