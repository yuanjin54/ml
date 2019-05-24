import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.get('data')
Y = iris.get('target')
# 对数据拆分，同时这个函数还会打乱数据顺序
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
print(Y)
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
print(knn.predict(X_test))
print(Y_test)
