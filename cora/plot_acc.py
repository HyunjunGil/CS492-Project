import numpy as np
import matplotlib.pyplot as plt



acc_louvain = np.load('train_acc_louvain_cora.npy')
acc_random = np.load('train_acc_random_cora.npy')

plt.plot(acc_louvain, color = 'red')
plt.plot(acc_random, color = 'blue')

plt.show()