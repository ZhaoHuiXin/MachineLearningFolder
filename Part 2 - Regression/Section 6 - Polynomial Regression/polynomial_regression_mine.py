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
# 线性回归模型，为了和多项式回归模型形成对比
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


## Fitting Polynomial Regression to the dataset
# 多项式回归模型， 会将自变量转化成，包含自变量不同次方的一个矩阵
from sklearn.preprocessing import PolynomialFeatures
# degree 多项式最高次数
poly_reg = PolynomialFeatures(degree=4)
# 多项式线性回归模型和多元线性回归模型类似，只不过参数是自变量X的不同次项
# 首先创建包含自变量不同次数的多项式矩阵,第0列是常数项
X_poly = poly_reg.fit_transform(X)
# 然后，构建多项式模型，创建方法和线性回归一样
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


## 对比线性回归模型和多项式回归模型的结果
## Visulising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth of Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


## Visulising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
# 使用 poly_reg.fit_transform(X) 更加方便
# plt.plot(X, lin_reg_2.predict(X_poly), color='green')
plt.title('Truth of Bluff (Polyminal Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# 提高多项式次数可以改进模型的拟合度，这里认为4次得到的模型最好


## 改进模型的平滑度————缩小自变量的间距
# np.arange得到在某一区间平均分布的值 min max sep
X_grid = np.arange(min(X), max(X), 0.1)
# 将X_grid转化成矩阵形式  rows cols
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
# 使用 poly_reg.fit_transform(X) 更加方便
# plt.plot(X, lin_reg_2.predict(X_poly), color='green')
plt.title('Truth of Bluff (Polyminal Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


## 使用拟合好的模型预测求职者报的实际薪水
# 使用多元线性回归模型预测
res1 = lin_reg.predict([[6.5]]) # >> 330378.78787879

# 使用多项式模型预测
res2 = lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) # >> 158862.45265153