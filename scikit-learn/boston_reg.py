from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

boston = load_boston()
X = boston.get('data')
Y = boston.get('target')
print(X.shape)
print(Y.shape)

# 对数据拆分，同时这个函数还会打乱数据顺序
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
model = LinearRegression()
model.fit(X_train, Y_train)
# W是训练出来的模型参数
W = model.coef_
print(model.intercept_)
print(W)
Y_pre = model.predict(X_test)
plt.figure()
plt.scatter(Y_test, Y_pre, 'r*')
plt.show()
print(np.power(metrics.mean_squared_error(Y_test, Y_pre), 0.5))
