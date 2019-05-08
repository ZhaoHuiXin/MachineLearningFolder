# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# 这里的数据是真是环境的模拟，并不是真实的数据，我们要得到的是强化学习这个方法
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
from math import sqrt, log
d = 10
# n轮前 每个广告 被选择的总次数
numbers_of_selections = [0] * d
# n轮前 每个广告 获得的奖励
sums_of_rewards = [0] * d
N = 10000
ads_selected = []
total_reward = 0
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
    # 每一轮选择的光告是哪个
    ads_selected.append(ad)
    # 获得本轮奖励
    reward = dataset.values[n, ad]
    # 更新numbers_of_selections, sums_of_rewards
    # 广告被选次数 + 1
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    # 加上本轮得分
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    # 执行本策略时针对所有广告的总奖励数
    total_reward = total_reward + reward
    
# Visualising the results
# 用柱状图表示每个广告被点击次数
# plt.hist画柱状图
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
