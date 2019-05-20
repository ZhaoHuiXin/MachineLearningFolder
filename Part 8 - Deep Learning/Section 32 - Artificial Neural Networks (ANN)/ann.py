# Artificial Neural Network

# Installing Tensorflow and Keras
 
# 1. On Mac: open "Terminal"
#    On Windows: open "Anaconda Prompt"  

# 2. Type:
# conda install tensorflow
# conda install -c conda-forge keras
# conda update --all


# Part 1 - Data Preprocessing
# Classification template

## 1.准备工作 数据预处理
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# X = dataset.iloc[:, 2:4].values
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
# Geography
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# translate variable to Domi variable when its val num more than 2
# here it's necessary to Geography, but not to Gender
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# avoid to drop Domi trap, we can del one col, eg col0
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling 特征缩放
# y的值要么0要么1，不需要特征缩放
# avoid input one variable  val is huge bigger than another.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Part 2 - Now let's make the ANN!
# Import the Keras libraries and packages
import keras
# use to  initial Neural Network package
from keras.models import Sequential
# help to add new layer to Neural Network
from keras.layers import Dense

# use Sequential initialsing ANN
classifier = Sequential()

# use Dense add the input layer and the first hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))

# adding the second hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))

# Adding the output layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


# Part 3 - Making the predictions and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)






