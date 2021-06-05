import torch
import numpy as np
import scipy.sparse as sp

from utils import encode_onehot, normalize, sparse_mx_to_torch_sparse_tensor, text_to_label_map, text_to_array

def load_cora(feature_mode = 0, feature_scale = 1):
  print('Loading cora dataset...')

  idx_features_labels = np.genfromtxt("cora.content",
                                      dtype=np.dtype(str))
  
  labels = encode_onehot(idx_features_labels[:, -1])

  idx = np.array(idx_features_labels[:,0], dtype=np.int32)
  idx_map = {j: i for i, j in enumerate(idx)}
  edges_unordered = np.genfromtxt("cora.cites", dtype=np.int32)
  edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                   dtype=np.int32).reshape(edges_unordered.shape)
  adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                      shape=(labels.shape[0], labels.shape[0]),
                      dtype=np.float32)
  
  adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
  adj = normalize (adj + sp.eye(adj.shape[0]))


  if (feature_mode == -1):
    features_given = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features_given = normalize(features_given)
    features_given = torch.FloatTensor(np.array(features_given.todense()))

    louvain_label_map = text_to_label_map('cora.label', ' ')
    tot_labels = len(set(louvain_label_map))
    features_louvain = np.zeros((len(idx_map),tot_labels * feature_scale))
    for i in range(len(idx_map)):
      for j in range(feature_scale):
        features_louvain[i, louvain_label_map[i] * feature_scale + j] = 1
    features_louvain = torch.FloatTensor(normalize(features_louvain))

    features = torch.cat((features_given, features_louvain), 1)
  elif (feature_mode == 0):
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
  elif (feature_mode == 1):
    louvain_label_map = text_to_label_map('cora.label', ' ')
    tot_labels = len(set(louvain_label_map))
    features = np.zeros((len(idx_map),tot_labels * feature_scale))
    for i in range(len(idx_map)):
      for j in range(feature_scale):
        features[i, louvain_label_map[i] * feature_scale + j] = 1
    features = torch.FloatTensor(normalize(features))
  elif (feature_mode == 2):
    features = torch.randn(len(idx_map), feature_scale)
    features = features / (features * features).sum(1).sqrt().reshape(len(idx_map), 1)
  elif (feature_mode == 3):
    features = torch.ones(len(idx_map), feature_scale) / feature_scale

  idx_train = range(140)
  idx_val = range(200, 500)
  idx_test = range(500, 1500)

  
  labels = torch.LongTensor(np.where(labels)[1])
  adj = sparse_mx_to_torch_sparse_tensor(adj)

  idx_train = torch.LongTensor(idx_train)
  idx_val = torch.LongTensor(idx_val)
  idx_test = torch.LongTensor(idx_test)

  return adj, features, labels, idx_train, idx_val, idx_test


def load_amazon(feature_scale = 1):
  print('Loading Amazon dataset')

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

  adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape = (num_node, num_node), dtype = np.float32)
  adj = adj + adj.T.multiply(adj.T) - adj.multiply(adj.T > adj)
  adj = normalize(adj + sp.eye(adj.shape[0]))
  adj = sparse_mx_to_torch_sparse_tensor(adj)
  
  train_size = num_node * 2 // 10
  train_val_size = num_node * 4 // 10

  np.random.shuffle(nodes)
  train_idx = nodes[:train_size]
  val_idx = nodes[train_size: train_val_size]
  test_idx = nodes[train_val_size:]

  louvain_label_map = text_to_label_map('amazon.label', ' ')
  tot_labels = len(set(louvain_label_map))
  features = np.zeros((num_node,tot_labels * feature_scale))
  for i in range(num_node):
    for j in range(feature_scale):
      features[i, louvain_label_map[i] * feature_scale + j] = 1
  features = torch.FloatTensor(normalize(features))


  labels = torch.LongTensor(np.where(labels)[1])

  train_idx = torch.LongTensor(train_idx)
  val_idx = torch.LongTensor(val_idx)
  test_idx = torch.LongTensor(test_idx)

  return adj, features, labels, train_idx, val_idx, test_idx
  

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
  adj = sparse_mx_to_torch_sparse_tensor(adj)

  idx_train = torch.LongTensor(idx_train)
  idx_val = torch.LongTensor(idx_val)
  idx_test = torch.LongTensor(idx_test)

  return adj, features, labels, idx_train, idx_val, idx_test
  
