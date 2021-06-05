import numpy as np

from utils import *

from louvain import apply_louvain
from sknetwork.clustering import Louvain, modularity

edges = text_to_array('cora.cites', '\t')


f_edge = open('com-amazon.ungraph.txt', 'r')
lines = f_edge.readlines()
edges = []
cnt = 0
max_idx = 0
for line in lines:
  line = line.lstrip().rstrip().split('\t')
  line = list(map(lambda x : int(x), line))
  edges.append(line)
  max_idx = max(max_idx, line[0], line[1])
f_edge.close()

labels = []
f_label = open('com-amazon.top297.cmty.txt', 'r')
lines = f_label.readlines()
for line in lines:
  line = line.strip().split(' ')
  labels.append(int(line[1]))
f_label.close()
labels = encode_onehot(labels)

present = [0 for _ in range(max_idx + 1)]
for edge in edges:
  present[edge[0]] += 1
  present[edge[1]] += 1

nodes = []
for i in range(len(present)):
  if (present[i] != 0):
    nodes.append(i)

node_map = {j : i for i, j in enumerate(nodes)}

nodes = [i for i in range(len(node_map))]
num_node = len(nodes)

edges = np.array(list(map(lambda x : [node_map[x[0]], node_map[x[1]]], edges)))
labels, label_cnt_map, q = apply_louvain(edges, num_node)
print(labels)

save_dict_as(label_cnt_map, 'amazon.labelcnt')
save_array_as(labels, 'amazon.label')