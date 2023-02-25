# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 11:48:05 2021

@author: Aakash
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv(r'retail.csv')

data = data.dropna()
data.columns = ['hang','heart','hold','light', 'white','na']

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labelencoder = LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')

data['hang'] = labelencoder.fit_transform(data['hang'])
hang_df = pd.DataFrame(enc.fit_transform(data[['hang']]).toarray())

data['heart'] = labelencoder.fit_transform(data['heart'])
heart_df = pd.DataFrame(enc.fit_transform(data[['heart']]).toarray())

data['hold'] = labelencoder.fit_transform(data['hold'])
hold_df = pd.DataFrame(enc.fit_transform(data[['hold']]).toarray())

data['light'] = labelencoder.fit_transform(data['light'])
light_df = pd.DataFrame(enc.fit_transform(data[['light']]).toarray())

data['white'] = labelencoder.fit_transform(data['white'])
white_df = pd.DataFrame(enc.fit_transform(data[['white']]).toarray())

data['na'] = labelencoder.fit_transform(data['na'])
na_df = pd.DataFrame(enc.fit_transform(data[['na']]).toarray())

heart_df.columns = list(range(124,389))
hold_df.columns = list(range(390, 640))
light_df.columns = list(range(641,881))
na_df.columns = list(range(882,1066))

data = data.join(hang_df)
data = data.join(heart_df)
data = data.join(hold_df)
data = data.join(light_df)
data = data.join(na_df)

del data['hang']
del data['heart']
del data['hold']
del data['light']
del data['white']
del data['na']

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
