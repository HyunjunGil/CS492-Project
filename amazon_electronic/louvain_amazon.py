import time
import numpy as np
import torch

from utils import *
import scipy.sparse as sp

from louvain import apply_louvain
from sknetwork.clustering import Louvain, modularity

loader = np.load('amazon_electronics_computer.npz')
loader = dict(loader)
adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

print(adj.shape)
louvain = Louvain()
labels = louvain.fit_transform(adj)
print(labels.shape)
labels_cnt = len(set(labels))

q = modularity(adj, labels)

save_array_as(labels, 'amazon_computer.label')

print('======== Louvain Method result ===========')
print('Total labels : {}'.format(labels_cnt))
print('Modularity for this label set : {}'.format(q))

loader = np.load('amazon_electronics_photo.npz')
loader = dict(loader)
adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

print(adj.shape)
louvain = Louvain()
labels = louvain.fit_transform(adj)
labels_cnt = len(set(labels))
print(labels.shape)

q = modularity(adj, labels)

save_array_as(labels, 'amazon_photo.label')

print('======== Louvain Method result ===========')
print('Total labels : {}'.format(labels_cnt))
print('Modularity for this label set : {}'.format(q))