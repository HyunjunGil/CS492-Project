import numpy as np
import scipy.sparse as sp
from sknetwork.clustering import Louvain, modularity

def apply_louvain(edge_list, num_node):
  edge_from = list(map(lambda x : x[0], edge_list)) + list(map(lambda x : x[1], edge_list))
  edge_to = list(map(lambda x : x[1], edge_list)) + list(map(lambda x : x[0], edge_list))
  values = [1 for i in range(len(edge_from))]
  adj = sp.csr_matrix((values, (edge_from, edge_to)), shape = (num_node, num_node))
  louvain = Louvain()
  labels = louvain.fit_transform(adj)
  labels_cnt = len(set(labels))
  label_cnt_map = {}
  for l in labels:
    if (l in label_cnt_map.keys()):
      label_cnt_map[l] += 1
    else:
      label_cnt_map[l] = 1
  q = modularity(adj, labels)

  print('======== Louvain Method result ===========')
  print('Total labels : {}'.format(labels_cnt))
  print('Modularity for this label set : {}'.format(q))
  print('Number of nodes for each label')
  print(label_cnt_map)

  return labels, label_cnt_map, q


