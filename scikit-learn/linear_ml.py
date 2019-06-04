from sklearn import datasets, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.make_regression(n_samples=200, n_features=1, n_targets=1, noise=1)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)
y_pre = model.predict(X_test)
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pre)))
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pre)
plt.show()
