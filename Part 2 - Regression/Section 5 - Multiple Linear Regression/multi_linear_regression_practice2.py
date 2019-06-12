#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:09:19 2019

@author: zhx
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,3] = LabelEncoder().fit_transform(X[:,3])
X = OneHotEncoder(categorical_features=[3]).fit_transform(X).toarray()
# avoid dummpy trap
X = X[:,1:]

## all in linearRegression
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)), values=X, axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_opt = X_train[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()