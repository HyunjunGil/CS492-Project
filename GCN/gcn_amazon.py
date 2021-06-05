import math
import time
import argparse
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
  def __init__(self, in_features, out_features, bias=True):
    super(GraphConvolution, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(torch.FloatTensor(in_features, out_features))
    if bias:
      self.bias = Parameter(torch.FloatTensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)
  ## input : n * k matrix where n is number of nodes and k is number of features
  def forward(self, input, adj):
    support = torch.mm(input, self.weight)
    output = torch.spmm(adj, support)
    if self.bias is not None:
      return output + self.bias
    else:
      return output

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GCN(nn.Module):
  def __init__(self, nfeat, nhid, nclass, dropout):
    super(GCN, self).__init__()
    self.gc1 = GraphConvolution(nfeat, nhid)
    self.gc2 = GraphConvolution(nhid, nclass)
    self.dropout = dropout

  def forward(self, x, adj):
    x = F.relu(self.gc1(x, adj))
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gc2(x,adj)
    return F.log_softmax(x, dim=1)

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
  f_edge.close()

  labels = []
  f_label = open('com-amazon.top297.cmty.txt', 'r')
  lines = f_label.readlines()
  for line in lines:
    line = line.strip().split(' ')
    labels.append(int(line[1]))
  f_label.close()
  labels = encode_onehot(labels)

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

  edges = np.array(list(map(lambda x : [node_map[x[0]], node_map[x[1]]], edges)))

  # edge_from = list(map(lambda x : node_map[x[0]], edges)) + list(map(lambda x : node_map[x[1]], edges))
  # edge_to = list(map(lambda x : node_map[x[1]], edges)) + list(map(lambda x : node_map[x[0]], edges))

  adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape = (num_node, num_node), dtype = np.float32)
  adj = adj + adj.T.multiply(adj.T) - adj.multiply(adj.T > adj)
  adj = normalize(adj + sp.eye(adj.shape[0]))
  adj = sparse_mx_to_torch_sparse_tensor(adj)

  # for edge in edges:
  #   adj[edge[0], edge[1]] = 1
  
  # adj = (adj + torch.eye(num_node))
  # adj = adj / adj.sum(1).reshape(num_node, 1)
  
  train_size = num_node * 5 // 10
  train_val_size = num_node * 8 // 10

  np.random.shuffle(nodes)
  train_idx = nodes[:train_size]
  val_idx = nodes[train_size: train_val_size]
  test_idx = nodes[train_val_size:]

  features = torch.ones(num_node, feature_size) / feature_size
  labels = torch.LongTensor(np.where(labels)[1])

  train_idx = torch.LongTensor(train_idx)
  val_idx = torch.LongTensor(val_idx)
  test_idx = torch.LongTensor(test_idx)

  return adj, features, labels, train_idx, val_idx, test_idx

def train(epoch):
    # 먼저 train모드로 epoch 한 번을 돌림
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args["fastmode"]:
        # 현재 idx_val부분까지도 train모드로 돌아간 상태이므로
        # 정확한 validation accuracy를 구하려면 전체를 다시 트레이닝 해주어야 한다.
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    # validation_loss 부분을 다시 계산해준다.
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    # 결과를 출력한다.
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

args = {
    "no_cuda": False,
    "fastmode": False,
    "seed": 42,
    "epochs": 200,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "hidden": 16,
    "dropout": 0.5
}

args["cuda"] = not args["no_cuda"] and torch.cuda.is_available()

np.random.seed(args["seed"])
torch.manual_seed(args["seed"])
if args["cuda"]:
    torch.cuda.manual_seed(args["seed"])

adj, features, labels, idx_train, idx_val, idx_test = load_data(300)
print(labels.sum(0))
print(labels)

model = GCN(nfeat=features.shape[1],
            nhid=args["hidden"],
            nclass=labels.max().item() + 1,
            dropout=args["dropout"])
optimizer = optim.Adam(model.parameters(),
                       lr=args["lr"], weight_decay=args["weight_decay"])

if args["cuda"]:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

t_total = time.time()
for epoch in range(200):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

test()