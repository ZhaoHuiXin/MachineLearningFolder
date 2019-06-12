#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:54:17 2019

@author: zhx
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,3].values

# impute missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
X[:,1:3] = imputer.fit_transform(X[:,1:3])

# dummpy variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,0] = LabelEncoder().fit_transform(X[:,0])
y = LabelEncoder().fit_transform(y)
X = OneHotEncoder(categorical_features=[0]).fit_transform(X).toarray()

# feature scaling
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
X = ss.fit_transform(X)

# divide train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

