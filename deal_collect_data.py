import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data/txt/collect.txt', header=None,
                   names=['duration', 'sizeAtDown', 'sizeAtUp', 'sizeAvg',
                          'pressureAtDown', 'pressureAtUp', 'pressureAvg'])
features = np.array(data)

plt.scatter(features[:9, 3], features[:9, 6], marker='o', c='r')
plt.scatter(features[10:, 3], features[10:, 6], marker='x', c='g')
plt.xlabel('Avg Size')
plt.ylabel('Avg Pressure')
plt.legend(('light press', 'heavy press'), loc='best', markerscale=1., numpoints=2, scatterpoints=1, fontsize=12)
plt.show()
