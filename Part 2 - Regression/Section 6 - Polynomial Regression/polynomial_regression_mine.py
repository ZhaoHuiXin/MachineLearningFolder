## Polynomial Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# 这里等级对应的就是不同的职位，两者是等价的；所以并不需要把职位包含在自变量里面
# X must be a matrix   
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# 这里不用分割训练集和测试集
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
# 这里也不用对数据集进行特征缩放
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
# 数据预处理到这儿就完成了


## Fitting Linear Regression to the dataset vs
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

## Fitting Polynomial Regression to the dataset
# 会将自变量转化成，包含自变量不同次方的一个矩阵
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()