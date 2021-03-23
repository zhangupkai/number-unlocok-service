import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

blobs = make_blobs(n_samples=200, n_features=2, random_state=1, centers=4)

X_blobs = blobs[0]  # 提取特征数据
print(X_blobs)

plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=blobs[1])
plt.show()

# 构建KMeans模型对象，设定k=4
k_means = KMeans(n_clusters=4)

# 训练模型
k_means.fit(X_blobs)

# 绘制可视化图
x_min, x_max = X_blobs[:, 0].min() - 0.5, X_blobs[:, 0].max() + 0.5
y_min, y_max = X_blobs[:, 1].min() - 0.5, X_blobs[:, 1].max() + 0.5

# 生成网格点矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='hermite', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.winter,
           aspect='auto', origin='lower')
plt.plot(X_blobs[:, 0], X_blobs[:, 1], 'w.', markersize=5)

# 红色x表示簇中心
centroids = k_means.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=3, color='r', zorder=10)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks()
plt.yticks()
plt.show()
