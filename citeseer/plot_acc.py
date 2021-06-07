import numpy as np
import matplotlib.pyplot as plt



acc_louvain = np.load('train_acc_louvain_cite.npy')
acc_random = np.load('train_acc_random_cite.npy')

plt.plot(acc_louvain, color = 'red')
plt.plot(acc_random, color = 'blue')

plt.show()