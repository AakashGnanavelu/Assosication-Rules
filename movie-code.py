# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:23:34 2021

@author: Aakash
"""


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv('movie.csv')

del data['V1']
del data['V2']
del data['V3']
del data['V4']
del data['V5']

from collections import Counter
item_frequencies = Counter(data)

item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

frequent_itemsets = apriori(data, min_support=0.008, max_len=3, use_colnames = True)

frequent_itemsets.sort_values('support', ascending = False, inplace=True)

import matplotlib.pyplot as plt

plt.bar(x = list(range(1,11)), height = frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold = 3)
rules.head(10)
rules.sort_values('lift', ascending = False, inplace=True)

def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

rules_no_redudancy  = rules.iloc[index_rules, :]

rules_no_redudancy.sort_values('lift', ascending=False).head(10)
