import pandas as pd
import numpy as np
import apriori

data = pd.read_csv("聚类结果.csv", encoding='gbk', usecols=[0, 1, 2, 7])
result = pd.DataFrame(np.zeros(shape=(20, 4)), columns=['规则', '支持度', '置信度', '提升度'])
for i, rule in enumerate(apriori.generate_rules(data.values.tolist(), min_support=0.1, min_confidence=0.25)):
    if i % 2 == 0:
        msg = (f'{rule.format_rule():20s}\t\t'
               f'(support={rule.support:0.4f}, confidence={rule.confidence:0.4f}, lift={rule.lift:0.4f})')
        print(msg)
        result.iloc[int(i / 2), 0] = rule.format_rule()
        result.iloc[int(i / 2), 1] = np.round(rule.support, 3)
        result.iloc[int(i / 2), 2] = np.round(rule.confidence, 3)
        result.iloc[int(i / 2), 3] = np.round(rule.lift, 3)
result.to_csv("apriori.csv")
