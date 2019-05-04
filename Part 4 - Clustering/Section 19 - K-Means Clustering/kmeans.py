# K-Means Clustering

# 1.Importing the libraries
import numpy as np # math methods
import matplotlib.pyplot as plt # draw
import pandas as pd # operate data


# 2.Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')

# X包含自变量的矩阵 iloc fetch some row and col
# ,左边代表取得行数，右边代表取得列数（除去最后一列）
X = dataset.iloc[:, 3:5].values

# 集群分析里面，因变量是不存在的，这类问题叫做“无监督学习”
# 训练集也不需要。特征缩放也不需要

## using hte elbow method find the optimal number of clusters
from sklearn.cluster import KMeans
# 创建一个空向量
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,max_iter=300,n_init=10, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("the elbow method")
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()
## 得出手肘拐点 5， 最佳组数
## 对数据进行最终的集群分析
## appling the k-means to the mall dataset
kmeans = KMeans(n_clusters=5,max_iter=300,n_init=10,
                init='k-means++', random_state=0)
## 用kmeans预测每一个顾客在哪个群组
# 首先将属于某一个群组的结果都放在一个向量里面y_kmeans
# 1.用数据拟合kmeans， 2. 用拟合好的kmeans算出对于每一个数据它所属的群组
y_kmeans = kmeans.fit_predict(X)

## 可视化集群结果
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster0')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster1')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster2')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster3')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster4')


plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1], s=300, c='black', label='Clusterids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1——100)')
plt.legend()
plt.show()
