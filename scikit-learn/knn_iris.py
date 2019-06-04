import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['class'] = iris.target
# df.loc[df['class'] == 0, 'class'] = iris.target_names[0]
df['class'] = df['class'].map({0: iris.target_names[0],
                               1: iris.target_names[1],
                               2: iris.target_names[2]}
                              )
x = iris.data
y = iris.target.reshape(-1, 1)
# 划分数据集，random_state表示随机划分的程度，stratify表示按照参考划分的列（这里的意思是y的每一种类型都要随机，不能全部偏向某一类型）
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50, stratify=y)


# 计算向量a与矩阵b的距离
def li_distance(a, b):
    return np.sum(np.abs(a - b), axis=1)


class KNN(object):
    def __init__(self, n_neighbors=1, dist_func=li_distance):
        self.n_neighbors = n_neighbors
        self.dist_func = li_distance

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)
        for i in range(len(x)):
            # 计算与x_train的距离
            distance = self.dist_func(x[i], self.x_train)
            # 从小到大排序,返回的是索引值
            dis = np.argsort(distance)
            # 讲dis映射到target
            n_y = self.y_train[dis[:self.n_neighbors]].ravel()
            # 获取出现次数最大的target
            y_pred[i] = np.argmax(np.bincount(n_y))
        return y_pred


# 开始测试
knn = KNN()
for k in range(1, 20, 2):
    knn.n_neighbors = k
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)

    # 求出预测准确率
    accuracy = accuracy_score(y_test, y_predict)
    print(accuracy)
