import time
import numpy as np
import torch

from utils import *

from louvain import apply_louvain
from sknetwork.clustering import Louvain, modularity

edges = text_to_array('cora.cites', '\t')

# idx_features_labels = np.genfromtxt("cora.content", dtype=np.dtype(str))
# nodes = idx_features_labels[:, 0]


f_content = open('cora.content', 'r')
lines = f_content.readlines()
label_map = []
feature_map = []
nodes = []
for line in lines:
  line = line.strip().split('\t')
  nodes.append(int(line[0]))
  feature_map.append(list(map(lambda x : int(x), line[1: -1])))
  label_map.append(line[-1])

node_map = {j : i for i, j in enumerate(nodes)}
print(node_map.items())
num_node = len(node_map)
print(num_node)
edges = list(map(lambda x : [node_map[x[0]], node_map[x[1]]], edges))
labels, label_cnt_map, q = apply_louvain(edges, num_node)
print(labels)

save_dict_as(label_cnt_map, 'cora.labelcnt')
save_array_as(labels, 'cora.label')

# ======== Louvain Method result ===========
# Total labels : 116
# Modularity for this label set : 0.8183546158465342


