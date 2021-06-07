from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('data/txt/scenes_collect.txt') as file:
    for line in file:
        result = list(line.strip('\n').split(','))
        result = list(map(int, result))
        print(result)
        duration_mean = np.mean(result[:4])
        print('duration_mean:', duration_mean)
        index = 4
        for duration in result[:4]:
            above_rate = (duration - duration_mean) / duration_mean
            training_file = open('data/txt/scenes_training_data.txt', 'a')
            training_file.writelines([str(above_rate), ',', str(result[index]), '\n'])
            training_file.close()
            index += 1

data = pd.read_csv('data/txt/scenes_training_data.txt', header=None)
data = np.array(data)

X_train = np.array(data[:-2, 0]).reshape(-1, 1)
y_train = data[:-2, 1]
X_test = np.array(data[-2:, 0]).reshape(-1, 1)
y_test = data[-2:, 1]

print(X_train.shape)
print(y_train.shape)

knn = KNeighborsClassifier()
# 调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签
knn.fit(X_train, y_train)
# 调用该对象的测试方法，主要接收一个参数：测试数据集
y_predict = knn.predict(X_test)
# 计算各测试样本基于概率的预测
probability = knn.predict_proba(X_test)
# 计算与最后一个测试样本距离在最近的5个点，返回的是这些样本的序号组成的数组
# neighbor_point = knn.kneighbors(np.array(X_test[-1]).reshape(1, -1), 5, False)
# 调用该对象的打分方法，计算出准确率
# score = knn.score(X_test, y_test, sample_weight=None)

print('y_predict: ', y_predict)
print('y_test: ', y_test)
# print('Accuracy: ', score)
# print('neighbor_point of last test sample:', neighbor_point)
print('probability: ', probability)
