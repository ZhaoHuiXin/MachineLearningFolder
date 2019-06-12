#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:31:11 2019

@author: zhx
"""

import numpy as np
import pandas as pd

# 2 read data
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

# 3.dummpy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,3] = LabelEncoder().fit_transform(X[:,3])
ohe = OneHotEncoder(categorical_features=[3])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]

## Method: all-in 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
all_in_regressor = LinearRegression()
all_in_regressor.fit(X_train, y_train)
y_pred = all_in_regressor.predict(X_test)

## Method: back elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)), values=X, axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_opt = X_train[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

# Adj. R-squared:                  0.947  best
X_opt = X_train[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

#  Adj. R-squared:                  0.944
X_opt = X_train[:,[0,3]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

