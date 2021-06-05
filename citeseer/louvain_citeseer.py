import time
import numpy as np
import torch

from utils import *

from louvain import apply_louvain
from sknetwork.clustering import Louvain, modularity

f_edge = open('citeseer.edges', 'r')
lines = f_edge.readlines()
edges = []
for line in lines:
  line = line.strip().split(',')
  line = list(map(lambda x : int(x) - 1, line))
  assert line[2] == 0
  edges.append([line[0], line[1]])
  edges.append([line[0], line[1]])

f_edge.close()

f_node = open('citeseer.node_labels', 'r')
lines = f_node.readlines()
nodes = [i for i in range(len(lines))]

labels, label_cnt_map, q = apply_louvain(edges, len(nodes))
print(labels)

save_dict_as(label_cnt_map, 'citeseer.labelcnt')
save_array_as(labels, 'citeseer.label')

# ======== Louvain Method result ===========
# Total labels : 422
# Modularity for this label set : 0.8888119762651289