# coding=utf-8

import os
import time
import random
import argparse
import numpy as np
import torch

from trainer import Trainer
from loader import load_gnn_data, load_gcn_graph
from models import SLGAT, SpSLGAT
import utils

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--pre-epochs', type=int, default=200, help='Number of epochs to pre-train.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--class-hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--input-dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight-dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='cora', help='Dataset name')
parser.add_argument('--file', type=str, default=None, help='Result file')
parser.add_argument('--cuda-id', type=int, default=None, help="CUDA id")
parser.add_argument('--model', type=str, default='SLGAT', help="Model name")
parser.add_argument('--tau', type=float, default=0.1, help="Tau")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    if args.cuda_id is not None:
        torch.cuda.set_device(args.cuda_id)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.determinstic = True
os.environ['PYTHONHASHSEED'] = str(args.seed)

# Load data
adj, features, labels, labels_onehot, idx_train, idx_val, idx_test = \
    load_gnn_data(args.data, sparse=True)

num_class = int(labels.max()) + 1
num_feature = features.size()[1]
idx_all = torch.LongTensor([i for i in range(features.size()[0])])
idx_unlabeled = torch.LongTensor([i for i in range(idx_train.size()[0], features.size()[0])])

print('# seed:', args.seed)
print('# nodes:', features.size()[0])
print('# features:', num_feature)
print('# classes:', num_class)
print('# model:', args.model)

if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    labels_onehot = labels_onehot.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()
    idx_unlabeled = idx_unlabeled.cuda()

# Model and optimizer
if args.model == 'SLGAT':
    gcn_adj = load_gcn_graph(args.data)
    if args.cuda:
        gcn_adj = gcn_adj.cuda()
    biased_adj = utils.adj_bias_normalize(adj.to_dense())
    model = SLGAT(num_feature=num_feature, hidden_dim=args.hidden, num_class=num_class,
                  class_hidden=args.class_hidden, adj=biased_adj, gcn_adj=gcn_adj.to_dense(),
                  input_dropout=args.input_dropout, dropout=args.dropout,
                  weight_dropout=args.weight_dropout)
    if args.cuda:
        model.cuda()
    trainer = Trainer(args, model)
elif args.model == 'SpSLGAT':
    model = SpSLGAT(num_feature=num_feature, hidden_dim=args.hidden, num_class=num_class,
                    class_hidden=args.class_hidden, adj=adj.to_dense(), gcn_adj=adj,
                    input_dropout=args.input_dropout, dropout=args.dropout,
                    weight_dropout=args.weight_dropout)
    if args.cuda:
        model.cuda()
    trainer = Trainer(args, model)
else:
    raise ValueError('invalid model name: {}'.format(args.model))

train_opt = {}
results = []


def get_accuracy(results):
    best_epoch = 0
    best_dev, acc_test = 0.0, 0.0
    for i, (d, t) in enumerate(results):
        if d > best_dev:
            best_epoch = i
            best_dev, acc_test = d, t
    return best_epoch, acc_test


def train():
    pre_epochs = args.pre_epochs
    epochs = args.epochs
    tau = args.tau

    if pre_epochs > 0:
        for i in range(pre_epochs):
            loss = trainer.update_soft(features, labels_onehot, idx_train)
            train_loss, _, train_acc = trainer.evaluate(features, labels, idx_train)
            _, _, val_acc = trainer.evaluate(features, labels, idx_val)
            _, _, test_acc = trainer.evaluate(features, labels, idx_test)

            print('pre-train Epoch: {:04d}'.format(i),
                  'train loss: {:.4f}'.format(train_loss),
                  'train acc: {:.4f}'.format(train_acc),
                  'val acc: {:.4f}'.format(val_acc),
                  'test acc: {:.4f}'.format(test_acc))

    for i in range(epochs):
        t = time.time()

        coef = random.randint(0, 1)

        targets = trainer.predict(features, tau=tau)
        targets[idx_train] = labels_onehot[idx_train]

        trainer.update_soft(features, targets, idx_train, idx_unlabeled=idx_unlabeled, coef=coef)

        train_loss, _, train_acc = trainer.evaluate(features, labels, idx_train)
        val_loss, _, val_acc = trainer.evaluate(features, labels, idx_val)
        _, _, test_acc = trainer.evaluate(features, labels, idx_test)
        results.append((val_acc, test_acc))

        print('Epoch: {:04d}'.format(i),
              'train loss: {:.4f}'.format(train_loss),
              'train acc: {:.4f}'.format(train_acc),
              'val loss: {:.4f}'.format(val_loss),
              'val acc: {:.4f}'.format(val_acc),
              'test acc: {:.4f}'.format(test_acc),
              'time: {:.4f}s'.format(time.time() - t))


t_total = time.time()

train()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

best_epoch, best_acc = get_accuracy(results)
print('{}th epoch: {:.4f}'.format(best_epoch, best_acc))

if args.file:
    with open(args.file, 'a') as f:
        f.write("{:.4f}\n".format(best_acc))
