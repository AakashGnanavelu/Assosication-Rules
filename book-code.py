# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:51:57 2021

@author: Aakash
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv('book.csv')

from collections import Counter
item_frequencies = Counter(data)

item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color='rgbkymc'); plt.xticks(list(range(0,11),), items[0:11]); plt.xlabel("items"); plt.ylabel("Count")

frequent_itemsets = apriori(data, min_support=0.008, max_len=3, use_colnames = True)

frequent_itemsets.sort_values('support', ascending = False, inplace=True)
plt.bar(x = list(range(1,11)), height = frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold = 5)
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

