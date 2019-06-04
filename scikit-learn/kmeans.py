from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# 生成data，200个样本，2个特征，中心点6个
x, y = datasets.make_blobs(n_samples=200, n_features=2, centers=6, cluster_std=1, random_state=100)


# plt.figure(figsize=(6, 4))
# plt.scatter(x[:, 0], x[:, 1], c=y)
# plt.show()

# 计算两个向量距离
def distance(a, b):
    return np.sum(np.square(a - b))


class Kmeans(object):
    # 构造方法
    def __init__(self, k):
        self.k = k

    def fit(self, x):
        # 迭代次数
        iterate_count = 20
        # 随机初始化聚簇分析
        self.cluster = x[np.random.randint(0, x.shape[0], self.k), :]
        while iterate_count > 0:
            # 对每一个样本x中的最小距离属于哪个簇
            x_cluster_index = []
            for i in range(x.shape[0]):
                dis = distance(self.cluster[0, :], x[i, :])
                cluster_index = 0
                # 迭代计算最小距离的簇
                for j in range(self.k):
                    if distance(self.cluster[j, :], x[i, :]) < dis:
                        cluster_index = j
                        dis = distance(self.cluster[j, :], x[i, :])
                x_cluster_index.append(cluster_index)
            # 更新cluster
            for j in range(self.k):
                x_index = []
                for ele in range(len(x_cluster_index)):
                    if j == x_cluster_index[ele]:
                        x_index.append(ele)
                self.cluster[j] = np.mean(x[x_index, :], axis=0)
            iterate_count -= 1

    def predict(self, x):
        x_cluster_index = []
        for i in range(x.shape[0]):
            dis = distance(self.cluster[0, :], x[i, :])
            index = 0
            for j in range(self.k):
                if distance(self.cluster[j, :], x[i, :]) < dis:
                    index = j
                    dis = distance(self.cluster[j, :], x[i, :])
            x_cluster_index.append(index)
        return np.array(x_cluster_index)


model = Kmeans(6)
model.fit(x)
y_pre = model.predict(x)
cluster = model.cluster
plt.figure(figsize=(6, 4))
plt.subplot(221)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title('原数据')

plt.subplot(222)
plt.scatter(x[:, 0], x[:, 1], c='r')
plt.scatter(cluster[:, 0], cluster[:, 1], c=np.array(range(6)), s=100)
plt.title('预测数据')
plt.show()
