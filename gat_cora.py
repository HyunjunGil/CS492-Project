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

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # softmax를 적용했을 때 거의 0에 가깝게 만들어 주려고 이렇게 설정함
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


def train(epoch, verbose=False):
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

    if verbose:
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

# ===============================================================================

torch.cuda.empty_cache()

args = {
    "no_cuda": False,
    "fastmode": False,
    "seed": 42,
    "epochs": 200,
    "lr": 0.005,
    "weight_decay": 5e-4,
    "hidden": 16,
    "dropout": 0.6,
    "nheads": 8,
    "alpha": 0.2,
    # feature_mode
    # -1 : Use given feature vector and Louvain initialized vector
    # 0 : Use given feature vector(word bag). In this case, FEATURE_SCALE will not be used
    # 1 : Use Louvain initialized vector scaled with FEATURE_SCALE
    # 2 : Use torch.randn(FEATURE_SCALE) as feature vector
    # 3 : Use torch.ones(FEATURE_SCALE) as feature vector(with normalization)
    "feature_mode": 0,
    "feature_scale": 1
}

args["cuda"] = not args["no_cuda"] and torch.cuda.is_available()

np.random.seed(args["seed"])
torch.manual_seed(args["seed"])
if args["cuda"]:
    torch.cuda.manual_seed(args["seed"])

adj, features, labels, idx_train, idx_val, idx_test = load_cora(args['feature_mode'], args['feature_scale'])
adj = adj.to_dense()

model = GAT(nfeat=features.shape[1],
            nhid=args["hidden"],
            nclass=labels.max().item() + 1,
            dropout=args["dropout"],
            alpha=args["alpha"],
            nheads=args["nheads"])
optimizer = optim.Adam(model.parameters(),
                       lr=args["lr"], weight_decay=args["weight_decay"])

if args["cuda"]:
    model = model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# =========================================================================================

t_total = time.time()
for epoch in range(args["epochs"]):
    train(epoch, verbose=(epoch % 10 == 0))
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


test()


# class GraphAttentionLayer(nn.Module):
#   def __init__(self, in_features, out_features, dropout, alpha, concat = True):
#     super(GraphAttentionLayer, self).__init__()
#     self.dropout = dropout
#     self.in_features = in_features
#     self.out_features = out_features
#     self.alpha = alpha
#     self.concat = concat

#     self.W = nn.Parameter(torch.empty(size = (in_features, out_features)))
#     nn.init.xavier_uniform_(self.W.data, gain = 1.414)
#     self.a = nn.Parameter(torch.empty(size = (2 * out_features, 1)))
#     nn.init.xavier_uniform_(self.a.data, gain = 1.414)

#     self.leakyrelu = nn.LeakyReLU(self.alpha)

#   def forward(self, h, adj):
#     Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
#     a_input = self._prepare_attentional_mechanism_input(Wh)
#     e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

#     # softmax를 적용했을 때 거의 0에 가깝게 만들어주려고 이렇게 설정함
#     # adj에 나와있지 않는 edge는 무시하기 위해서
#     zero_vec = -9e15*torch.ones_like(e)
#     attention = torch.where(adj > 0, e, zero_vec)
#     attention = F.softmax(attention, dim = 1)
#     attention = F.dropout(attention, self.dropout, training = self.training)
#     h_prime = torch.matmul(attention, Wh)

#     if self.concat:
#       return F.elu(h_prime)
#     else:
#       return h_prime

#   def _prepare_attentional_mechanism_input(self, Wh):
#     N = Wh.size()[0]

#     Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim = 0)
#     Wh_repeated_in_alternating = Wh.repeat(N, 1)

#     all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_in_alternating], dim = 1)

#     return all_combinations_matrix.view(N, N, 2 * self.out_features)

#   def __repr__(self):
#     return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'

# class GAT(nn.Module):
#   def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#     super(GAT, self).__init__()
#     self.dropout = dropout

#     self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout = dropout, alpha = alpha, concat = True) for _ in range(nheads)]
#     for i, attention in enumerate(self.attentions):
#       self.add_module('attention_{}'.format(i), attention)

#     self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout = dropout, alpha = alpha, concat = False)
