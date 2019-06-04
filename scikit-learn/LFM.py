# 引语义模型的梯度下降算法实现（LFM）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R = np.array([
    [4, 0, 2, 0, 1],
    [0, 2, 3, 0, 0],
    [1, 0, 2, 4, 0],
    [5, 0, 0, 3, 1],
    [0, 0, 1, 5, 1],
    [0, 3, 2, 4, 1]
])
print(R.shape)

'''
@ input params:
R: mxn的评分矩阵
K: 隐特征向量个数
max_iter: 最大迭代次数
alpha: 学习率
lamda: 正则化系数

@输出:
分解之后的P,Q
P: 初始化用户特征矩阵MxK
Q: 初始化物品特征矩阵NxK
'''

# 给定超参数取值
K = 3
max_iter = 1000
alpha = 0.001
lamda = 0.005


# 核心算法
def LFM_grad_desc(R, K=2, max_iter=100, alpha=0.0001, lamda=0.):
    M = R.shape[0]
    N = len(R[0])

    # 初始化Q and P
    P = np.random.rand(M, K)
    Q = np.random.rand(N, K).T
    loss = 0

    for step in range(max_iter):
        for u in range(M):
            for i in range(N):
                if R[u][i] > 0:
                    eui = np.dot(P[u, :], Q[:, i]) - R[u, i]
                    for k in range(K):
                        P[u][k] = P[u][k] - alpha * (2 * eui * Q[k][i] + 2 * lamda * P[u][k])
                        Q[k][i] = Q[k][i] - alpha * (2 * eui * P[u][k] + 2 * lamda * Q[k][i])
        loss = 0
        for u in range(M):
            for i in range(N):
                if R[u][i] > 0:
                    loss += (np.dot(P[u, :], Q[:, i]) - R[u][i]) ** 2
                    for k in range(K):
                        loss += lamda * (P[u][k] ** 2 + Q[k][i] ** 2)
        if loss < 0.0001:
            break
    return P, Q.T, loss


# loss_list = []
# for k in range(10):
#     P, Q, loss = LFM_grad_desc(R, k, max_iter, alpha, lamda)
#     loss_list.append(loss)
# plt.plot(range(10), loss_list)
# plt.show()

P, Q, loss = LFM_grad_desc(R, K, max_iter, alpha, lamda)
print(P)
print(Q)
print(loss)
preR = np.dot(P, Q.T)
print(preR)
print(R)
