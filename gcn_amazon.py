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
from dataset import load_amazon
from utils import accuracy

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
    "hidden": 300,
    "dropout": 0.5
}

args["cuda"] = not args["no_cuda"] and torch.cuda.is_available()

np.random.seed(args["seed"])
torch.manual_seed(args["seed"])
if args["cuda"]:
    torch.cuda.manual_seed(args["seed"])

adj, features, labels, idx_train, idx_val, idx_test = load_amazon(2)

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