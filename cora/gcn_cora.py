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

from dataset import load_cora
from utils import accuracy

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
    self.weight.data.uniform_(stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)
  
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
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args["fastmode"]:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val, acc_val

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    pred = output[idx_test].argmax(dim=1)[:, None].cpu()
    label = labels[idx_test].cpu()
    f1_scores = f1_score(label, pred, zero_division=1, average=None)
    print('F1 scores')
    for i, score in enumerate(f1_scores):
        print(f'Class {i}: {score:.2f}')

    cm = confusion_matrix(label, pred)
    cm = cm / cm.sum(axis=1)[:, None]
    _, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=np.array([0, 1, 2, 3, 4, 5, 6]))
    disp = disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal',
                    values_format='.2f')
    plt.show()

#=====================================================================================

args = {
    "no_cuda": False,
    "fastmode": False,
    "seed": 40,
    "epochs": 200,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "hidden": 500,
    "dropout": 0.5,
    # feature_mode
    # -1 : Use given feature vector and Louvain initialized vector
    # 0 : Use given feature vector(word bag). In this case, FEATURE_SCALE will not be used
    # 1 : Use Louvain initialized vector scaled with FEATURE_SCALE
    # 2 : Use torch.randn(FEATURE_SCALE) as feature vector(with L2 normalization)
    # 3 : Use torch.ones(FEATURE_SCALE) as feature vector(with L1 normalization)
    "feature_mode": 2, 
    "feature_scale": 1508
}

args["cuda"] = not args["no_cuda"] and torch.cuda.is_available()

np.random.seed(args["seed"])
torch.manual_seed(args["seed"])
if args["cuda"]:
    torch.cuda.manual_seed(args["seed"])

adj, features, labels, idx_train, idx_val, idx_test = load_cora(args["feature_mode"], args["feature_scale"])
print(features.size())
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

"""Start training."""

t_total = time.time()
loss_values = []
acc_values = []
for epoch in range(args["epochs"]):
  loss_val, acc_val = train(epoch)
  loss_values.append(loss_val.cpu().detach())
  acc_values.append(acc_val.cpu().detach())


np.save('train_acc_random_cora', acc_values)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

"""You can see the loss and accuracy of the model."""

test()