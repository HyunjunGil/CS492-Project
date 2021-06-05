import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *

from louvain import apply_louvain
from sknetwork.clustering import modularity
import scipy.sparse as sp


def load_data(feature_size):
  print('Loading Amazon dataset')

  f_edge = open('com-amazon.ungraph.txt', 'r')
  lines = f_edge.readlines()
  edges = []
  cnt = 0
  max_idx = 0
  for line in lines:
    if (cnt >= 4):
      line = line.lstrip().rstrip().split('\t')
      line = list(map(lambda x : int(x), line))
      edges.append(line)
      max_idx = max(max_idx, line[0], line[1])
    cnt += 1
  
  print(len(edges))

  f_edge.close()

  labels = []
  f_label = open('com-amazon.top297.cmty.txt', 'r')
  lines = f_label.readlines()
  for line in lines:
    line = line.strip().split(' ')
    labels.append(int(line[1]))
  
  f_label.close()

  present = [0 for i in range(max_idx + 1)]
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

  print(len(labels), num_node)

  edge_from = list(map(lambda x : node_map[x[0]], edges)) + list(map(lambda x : node_map[x[1]], edges))
  edge_to = list(map(lambda x : node_map[x[1]], edges)) + list(map(lambda x : node_map[x[0]], edges))
  
  print('1')

  adj = sp.csr_matrix(([1 for i in range(len(edge_from))], (edge_from, edge_to)), shape = (num_node, num_node))
  
  print('2')

  train_size = num_node * 5 // 10
  val_size = num_node * 8 // 10 - num_node * 5 // 10
  test_size = num_node - num_node * 8 // 10

  print('3')

  # train_val_idx = np.random.choice(nodes, train_size + val_size, replace = False)
  # test_idx = list(filter(lambda x : not x in train_val_idx, nodes))
  # train_idx = np.random.choice(train_val_idx, train_size, replace = False)
  # train_idx = list(filter(lambda x : x in train_idx, train_val_idx))
  # val_idx = list(filter(lambda x : not x in train_idx, train_val_idx))

  print('4')

  features = torch.ones(num_node, feature_size) / feature_size

  print('5')

  # train_idx = torch.LongTensor(train_idx)
  # val_idx = torch.LongTensor(val_idx)
  # test_idx = torch.LongTensor(test_idx)

  return adj, features, labels #, train_idx, val_idx, test_idx

# , train_idx, val_idx, test_idx 
adj, features, labels= load_data(500)
print('6')
print(modularity(adj, np.array(labels)))


# ####################################### Load file #######################################
# file_edges = open('com-amazon.ungraph.txt', 'r')
# cnt = 0
# edges = []
# max_idx = 0
# min_idx = 9999
# lines = file_edges.readlines()
# for line in lines:
#   cnt += 1
#   if (cnt < 5):
#     print(line)
#   if (cnt >= 5):
#     line = line.lstrip().rstrip().split('\t')
#     line = list(map(lambda x : int(x), line))
#     edges.append(line)
#     max_idx = max(max_idx, line[0], line[1])
#     min_idx = min(min_idx, line[0], line[1])


# print('done', 'extracting edge info')
# file_edges.close()

# present = [0 for i in range(max_idx + 1)]
# for edge in edges:
#   present[edge[0]] += 1
#   present[edge[1]] += 1

# nonzero_node  = []
# for i in range(len(present)):
#   if (present[i] != 0):
#     nonzero_node.append(i)

# print('len(nonzero_node) : ', len(nonzero_node))

# # Relabel each node from 1 to number of nodes
# node_label_map = {}
# for i in range(len(nonzero_node)):
#   node_label_map[nonzero_node[i]] = i

# num_node = len(node_label_map)
# num_edge = len(edges)

# edges = list(map(lambda x : [node_label_map[x[0]], node_label_map[x[1]]], edges)) + list(map(lambda x : [node_label_map[x[1]], node_label_map[x[0]]], edges))

# edge_weights = {}

# for edge in edges:
#   edge_weights[(edge[0], edge[1])] = 1

# print(len(edges), len(edge_weights))

# file_community= open('com-amazon.top5000.cmty.txt', 'r')

# lines = file_community.readlines()

# cmty = {}
# for i, line in enumerate(lines):
  
#   line = line.lstrip().rstrip().split('\t')
  
#   line = list(map(lambda x : node_label_map[int(x)], line))
#   cmty[i] = line

# num_cmty = len(cmty)
# total_cmty_nodes = 0
# cmty_size = []

# for i in range(num_cmty):
#   cmty_size.append(len(cmty[i]))
#   total_cmty_nodes += len(cmty[i])

# file_community.close()
# for i in range(10):
#   print(cmty[i])
# print(len(cmty), total_cmty_nodes, max(cmty_size), min(cmty_size))

# initial_cmty_info = {i : i for i in range(num_node)}
# #####################################################################

# cmty_info, q = apply_louvain(edges, size = (num_node, num_node))

# cmty_dict = {}

# for i in range(len(cmty_info)):
#   if (cmty_info[i] in cmty_dict.keys()):
#     cmty_dict[cmty_info[i]] += 1
#   else:
#     cmty_dict[cmty_info[i]] = 1

# for k, v in cmty_dict.items():
#   print('{} : {}'.format(k, v))

# f_label = open('com-amazon..top297.cmty.txt', 'w')
# for i in range(len(cmty_info)):
#   data = '{} {}\n'.format(i, cmty_info[i])
#   f_label.write(data)
# f_label.close()