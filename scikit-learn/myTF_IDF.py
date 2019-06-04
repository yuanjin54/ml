import math
import numpy as np
import pandas as pd

s_1 = "The cat sat on my bed"
s_2 = "The dog sat on my knees"

list_1 = s_1.split(' ')
list_2 = s_2.split(' ')

# 所有文章的出现的词
set_all = set(list_1).union(set(list_2))
list_all = [list_1, list_2]


# 计算文章i每个词的词频
def get_TF(list_i, set_all):
    set_i = set_all
    dict_i = dict.fromkeys(set_i, 0)
    for key in list_i:
        dict_i[key] += 1
    for k, v in dict_i.items():
        dict_i[k] = v / len(list_i)
    return dict_i


# 计算IDF
def get_IDF(set_all, list_all):
    list_size = len(list_all)
    dict_all = dict.fromkeys(set_all, 0)
    for key in dict_all.keys():
        for l in list_all:
            if key in l:
                dict_all[key] += 1
        dict_all[key] = math.log10((list_size + 1) / (dict_all[key] + 1))
    return dict_all


# 计算TF-IDF
def get_TFIDF(tf, idf):
    tfidf = {}
    for word in tf.keys():
        tfidf[word] = tf[word] * idf[word]
    return tfidf


tfidf1 = get_TFIDF(get_TF(list_1, set_all), get_IDF(set_all, list_all))
tfidf2 = get_TFIDF(get_TF(list_2, set_all), get_IDF(set_all, list_all))

print(pd.DataFrame([tfidf1, tfidf2]))
