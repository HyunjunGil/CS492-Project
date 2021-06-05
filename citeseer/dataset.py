import torch
import numpy as np
import scipy.sparse as sp

from utils import encode_onehot, normalize, sparse_mx_to_torch_sparse_tensor, text_to_label_map, text_to_array

def load_citeseer(feature_mode, feature_scale):
  print('Loading cora dataset...')

  f_edge = open('citeseer.edges', 'r')
  lines = f_edge.readlines()
  edges = []
  for line in lines:
    line = line.strip().split(',')
    line = list(map(lambda x : int(x) - 1, line))
    assert line[2] == 0
    edges.append([line[0], line[1]])
  f_edge.close()

  edges = np.array(edges)

  f_node = open('citeseer.node_labels', 'r')
  lines = f_node.readlines()
  labels = []
  for line in lines:
    line = line.strip().split(',')
    labels.append(int(line[1]))

  nodes = [i for i in range(len(lines))]
  num_node = len(nodes)
  
  labels = encode_onehot(labels)

  adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                      shape=(labels.shape[0], labels.shape[0]),
                      dtype=np.float32)
  
  adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
  adj = normalize (adj + sp.eye(adj.shape[0]))


  if (feature_mode == 0):
    features = torch.randn(num_node, feature_scale)
    features = features / (features * features).sum(1).sqrt().reshape(num_node, 1)
  elif (feature_mode == 1):
    features = torch.ones(num_node, feature_scale) / feature_scale
  elif (feature_mode == 2):
    louvain_label_map = text_to_label_map('citeseer.label', ' ')
    tot_labels = len(set(louvain_label_map))
    features = np.zeros((num_node ,tot_labels * feature_scale))
    for i in range(num_node):
      for j in range(feature_scale):
        features[i, louvain_label_map[i] * feature_scale + j] = 1
    features = torch.FloatTensor(normalize(features))
  

  train_size = num_node * 2 // 10
  train_val_size = num_node * 4 // 10
  np.random.shuffle(nodes)
  idx_train = nodes[:train_size]
  idx_val = nodes[train_size: train_val_size]
  idx_test = nodes[train_val_size:]

  
  labels = torch.LongTensor(np.where(labels)[1])
  print(labels)
  adj = sparse_mx_to_torch_sparse_tensor(adj)

  idx_train = torch.LongTensor(idx_train)
  idx_val = torch.LongTensor(idx_val)
  idx_test = torch.LongTensor(idx_test)

  return adj, features, labels, idx_train, idx_val, idx_test
  
