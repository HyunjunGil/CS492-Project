import torch
import numpy as np
import scipy.sparse as sp

from utils import encode_onehot, normalize, sparse_mx_to_torch_sparse_tensor, text_to_label_map

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