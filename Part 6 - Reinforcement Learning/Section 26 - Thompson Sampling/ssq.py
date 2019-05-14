# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ssq.csv')

# Implementing Thompson Sampling
import random
N = 2412
d = 16

# 普通列每个号码1-33获得奖励1的总次数
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
no_selected = []
total_reward = 0
col = 1+7
for n in range(0, N):
    no = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1,
                                         numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            no = i
    # 每一期该列选择的数字是哪个
    no_selected.append(no)
    # 获得本轮奖励，对于1列，如果记录的数与本轮 选择的相同 reward=1，否则reward=0
    real_no = dataset.values[n, col]
    if not real_no.isdigit():
        continue
    reward = 0
    if int(real_no) == no+1:
        reward = 1
        numbers_of_rewards_1[no] += 1
    else:
        numbers_of_rewards_0[no] += 1
    # 执行本策略时针对所有广告的总奖励数
    total_reward = total_reward + reward
print(total_reward)
map_1 = {i+1:numbers_of_rewards_1[i] for i in range(len(numbers_of_rewards_1))}
print(map_1)
print(max(map_1))
print(numbers_of_rewards_0)
# Visualising the results - Histogram
plt.hist(no_selected,bins=100)
plt.title('第一位选择柱状图')
plt.xlabel('no for selected')
plt.ylabel('Number of times each no was selected')
plt.show()