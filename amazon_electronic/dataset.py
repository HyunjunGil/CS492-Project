import torch
import numpy as np
import scipy.sparse as sp

from utils import encode_onehot, normalize, sparse_mx_to_torch_sparse_tensor, text_to_label_map, text_to_array

def load_amazon(feature_mode = 0, feature_scale = 1, dataset = 'computer'):
  print('Loading amazon-{} dataset...'.format(dataset))

  loader = np.load('amazon_electronics_{}.npz'.format(dataset))

  loader = dict(loader)
  adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])
                                

  if 'attr_data' in loader:
    # Attributes are stored as a sparse CSR matrix
    attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                shape=loader['attr_shape'])
  elif 'attr_matrix' in loader:
    # Attributes are stored as a (dense) np.ndarray
    attr_matrix = loader['attr_matrix'] 
  else:
    attr_matrix = None
  attr_matrix = torch.FloatTensor(np.array(attr_matrix.todense()))

  if 'labels_data' in loader:
    # Labels are stored as a CSR matrix
    labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                            shape=loader['labels_shape'])
  elif 'labels' in loader:
    # Labels are stored as a numpy array
    labels = loader['labels']
  else:
    labels = None
  labels = encode_onehot(np.array(labels))

  num_node = labels.shape[0]
  nodes = [i for i in range(num_node)]

  if (feature_mode == -1):
    louvain_label_map = text_to_label_map('amazon_{}.label'.format(dataset), ' ')
    tot_labels = len(set(louvain_label_map))
    features_louvain = np.zeros((num_node, tot_labels * feature_scale))
    for i in range(num_node):
      for j in range(feature_scale):
        features_louvain[i, louvain_label_map[i] * feature_scale + j] = 1
    features_louvain = torch.FloatTensor(normalize(features_louvain))
    features = torch.cat((attr_matrix, features_louvain), dim = 1)
  elif (feature_mode == 0):
    features = attr_matrix
  elif (feature_mode == 1):
    louvain_label_map = text_to_label_map('amazon_{}.label'.format(dataset), ' ')
    tot_labels = len(set(louvain_label_map))
    features_louvain = np.zeros((num_node, tot_labels * feature_scale))
    for i in range(num_node):
      for j in range(feature_scale):
        features_louvain[i, louvain_label_map[i] * feature_scale + j] = 1
    features = torch.FloatTensor(normalize(features_louvain))
  elif (feature_mode == 2):
    features = torch.randn(num_node, feature_scale)
    features = features / (features * features).sum(1).sqrt().reshape(num_node, 1)
  elif (feature_mode == 3):
    features = torch.ones(num_node, feature_scale) / feature_scale
    
  
  train_size = num_node * 2 // 10
  train_val_size = num_node * 4 // 10
  np.random.shuffle(nodes)
  idx_train = nodes[:train_size]
  idx_val = nodes[train_size: train_val_size]
  idx_test = nodes[train_val_size:]

  
  labels = torch.LongTensor(np.where(labels)[1])
  print(labels)
  adj = sparse_mx_to_torch_sparse_tensor(adj_matrix)
  print(adj)
  print(features)

  idx_train = torch.LongTensor(idx_train)
  idx_val = torch.LongTensor(idx_val)
  idx_test = torch.LongTensor(idx_test)

  return adj, features, labels, idx_train, idx_val, idx_test