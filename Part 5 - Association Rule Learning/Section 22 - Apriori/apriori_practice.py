#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:43:14 2019

@author: zhaohuixin
"""

import numpy as np
import pandas as pd

from apyori import apriori

dataset = pd.read_csv("Market_Basket_Optimisation.csv",header=None)
transactions = []
for i in range(7501):
    transactions.append([str(dataset.values[i,j]) for j in range(20)])
    
rules = apriori(transactions=transactions, min_length=2, min_support=3*7/7500,
                min_confidence=0.2, min_lift=3)

results = list(rules)

myResults = [list(x) for x in results]