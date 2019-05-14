# Natural Language Processing

#### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#### Importing the dataset 1. 读取数据
# quoting=3 去除引号
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
ceshi_set = pd.read_csv('ceshi.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts 2. 初始清理 去除所有标点符号和数字,并用空格替代（建立词袋）
import re
corpus = []
ceshi_corpus = []
for i in range(6):
    #review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = re.sub('[^a-zA-Z]',' ', ceshi_set['Review'][i])
    # 3. 将所有字母转为小写
    review = review.lower()
    # 4. 清理所有虚词，下载nltk字典并载入
    review = review.split()
    # import nltk
    # nltk.download('stopwords')
    from nltk.corpus import stopwords # 词袋模型
    # list python 搜索缓慢，可将 stopwords 转为set
    # review = [word for word in review if not word in set(stopwords.words('english'))]
    # 5. 词根化(中文中不存在词根化，主要是字母拼成的语言，甚至日文中也有词根化)
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # 6.字符串转化
    review = ' '.join(review)
    #corpus.append(review)
    ceshi_corpus.append(review)
    
#### Creating the Bag of Words model
# 8. 稀疏矩阵转换
from sklearn.feature_extraction.text import CountVectorizer
# 9. 最大过滤
cv = CountVectorizer(max_features = 1500)
# 拟合并转换
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
# 使用拟合好的cv，将要预测的数据转换成相同的维度
ceshi_X = cv.transform(ceshi_corpus).toarray()
# 10 最后一步，构造分类模型
#### Splitting the dataset into the Training set and Test set

#### Fitting Naive Bayes to the Training set

#### Predicting the Test set results




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling 特征缩放
# 这里不需要特征缩放


## 2. 创建分类器，并用测试集进行拟合训练
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

## 3.用拟合好的分类器预测测试集的结果——Predicting the Test set results
y_pred = classifier.predict(X_test)
ceshi_y = classifier.predict(ceshi_X)

## 4.用混淆矩阵评估分类器的性能——Making the Confusion Matrix
# import func confusion_matrix
from sklearn.metrics import confusion_matrix
# y_true:实实在在的真实的分类——Ground truth，对应y_test
# y_pred:预测出的结果——Estimated targets，对应y_pred
#### Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)