import torch
import numpy as np
import scipy.sparse as sp

def text_to_dict(file_name, divisor = ' '):
  f = open(file_name, 'r')
  lines = f.readlines()
  result = {}
  for line in lines:
    line = line.strip().split(divisor)
    line = list(map(lambda x : int(x), line))
    result[line[0]] = line[1]
  f.close()
  return result

def text_to_array(file_name, divisor = ' '):
  f = open(file_name, 'r')
  lines = f.readlines()
  result = []
  for line in lines:
    line = line.strip().split(divisor)
    line = list(map(lambda x : int(x), line))
    result.append(line)
  f.close()
  return result

def text_to_label_map(file_name, divisor = ' '):
  f = open(file_name, 'r')
  line = f.readline()
  line = line.strip().split(divisor)
  return list(map(lambda x : int(x), line))

def save_dict_as(d, name):
  f = open(name, 'w')
  for k, v in d.items():
    f.write('{} {}\n'.format(k, v))
  f.close()

def save_array_as(arr, name):
  f = open(name, 'w')
  for x in arr:
    f.write(str(x) + ' ')
  f.close()
    
def encode_onehot(labels):
  classes = set(labels)
  classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                  enumerate(classes)}
  labels_onehot = np.array(list(map(classes_dict.get, labels)),
                           dtype=np.int32)
  return labels_onehot

def normalize(mx):
  rowsum = np.array(mx.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0
  r_mat_inv = sp.diags(r_inv)
  mx = r_mat_inv.dot(mx)
  return mx

def accuracy(output, labels):
  preds = output.max(1)[1].type_as(labels)
  correct = preds.eq(labels).double()
  correct = correct.sum()
  return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
  sparse_mx = sparse_mx.tocoo().astype(np.float32)
  indices = torch.from_numpy(
      np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
  values = torch.from_numpy(sparse_mx.data)
  shape = torch.Size(sparse_mx.shape)
  return torch.sparse.FloatTensor(indices, values, shape)


