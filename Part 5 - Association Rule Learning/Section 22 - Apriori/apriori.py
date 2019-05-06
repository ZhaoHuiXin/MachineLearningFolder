# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
# min support = 3*7/7500 = 0.003
rules = apriori(transactions, min_support = 0.003,
                min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
# R中根据lift提升度排序
# python已经根据相关度排好序了，它是根据Support confidence lift一起制定的
results = list(rules)
myResults = [list(x) for x in results]
