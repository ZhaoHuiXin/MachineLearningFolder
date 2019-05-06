# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, log

# Importing the dataset
# 这里的数据是真是环境的模拟，并不是真实的数据，我们要得到的是强化学习这个方法
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
d = 10
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
N = 10000
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = sqrt(3/2 * log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
# Visualising the results
