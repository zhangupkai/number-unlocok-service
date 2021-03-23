# k-means 聚类
from sklearn.cluster import KMeans
import joblib
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.txt', header=None, names=['Duration'])
data.insert(1, 'Other', 1)
print(data.head())

# data.plot(kind='scatter', x='Duration', y='none', figsize=(12, 8))
X = np.array(np.mat(data))
print(X)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

k_means = KMeans(n_clusters=2)
k_means.fit(X)

# 保存模型
joblib.dump(k_means, 'cluster.pkl')

# 绘制可视化图
# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# 生成网格点矩阵
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
#
# Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='hermite', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.winter,
#            aspect='auto', origin='lower')
# plt.plot(X[:, 0], X[:, 1], 'w.', markersize=5)
# print(X[:, 0])
#
# # 红色x表示簇中心
# centroids = k_means.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=3, color='r', zorder=10)
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks()
# plt.yticks()
# plt.show()

print(k_means.predict(np.mat([[210, 1], [43, 1], [1, 1]])))
