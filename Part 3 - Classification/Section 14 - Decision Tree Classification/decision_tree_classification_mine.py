# Decision Tree Classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
# X = dataset.iloc[:, 2:4].values
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling 特征缩放
# 这里做的是逻辑回归，y的值要么0要么1，不需要特征缩放
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting Decision Tree to the Training set
# Create your classifier here
# criterion: entropy熵  or  gin
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)

## 2. 创建分类器，并用测试集进行拟合训练
# from sklearn.linear_model import ？
# random_state 确定随机数组的生成方式
# classifier = ？
# 用训练集拟合分类器，拟合过程中，分类器会逐渐学习X_train和y_train之间的相关度
classifier.fit(X_train, y_train)

## 3.用拟合好的分类器预测测试集的结果——Predicting the Test set results
y_pred = classifier.predict(X_test)

## 4.用混淆矩阵评估分类器的性能——Making the Confusion Matrix
# import func confusion_matrix
from sklearn.metrics import confusion_matrix
# y_true:实实在在的真实的分类——Ground truth，对应y_test
# y_pred:预测出的结果——Estimated targets，对应y_pred
cm = confusion_matrix(y_test, y_pred)

'''
cm
Out[15]: 
array([[65,  3],
       [ 8, 24]])
65和24对应正确预测的个数，8和3对应错误预测的个数；正确率(65+24)/100=89%
这是评估分类器性能的第一步，用混淆矩阵看有多少组正确预测，有多少组错误预测
8 代表真实样品结果为1，预测结果为0的样品
3 代表预测结果为1，实际结果为0的样品个数
'''

## 5.Visualing the Training set results
# ListedColormap 帮助我们给不同的点上不同的色
from matplotlib.colors import ListedColormap
# 为了防止频繁的改数据名，新建2个变量
X_set, y_set = X_train, y_train
# X1,X2 对应图中的像素点
# np.meshgrid，第一行取age行最小值减1，最大值加1，目的让图像边缘有留白，方便更清楚的看到生成的图像
# 每个像素点之间的横向距离step是0.01；第一行选出了对应所有年龄的网点的值
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
 # 第二行取对应年薪的最小值减1，最大值加1，目的也是更清楚的看到生成的图像
 # 每个像素点之间的纵向距离step是0.01
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# 将不同的像素点涂色；用已经分类好的分类器(线性)来预测每一个点所属的分类；并根据分类值涂不同的颜色
# 线性分类器的预测边界是一条直线
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# 标注最大值最小值
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# 画出实际观测的点，橙色和蓝色的
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j)
plt.title('Decision Tree (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
# 显示不同的点对应的值——图注
plt.legend()
# 生成图像
plt.show()

## 6.Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j)
plt.title('Decision Tree (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()