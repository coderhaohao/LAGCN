from __future__ import division
from __future__ import print_function

import time
import argparse
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    help='Name of dataset.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--modified', action='store_true', default=False,
                    help='Whether use the modified graph.')
parser.add_argument('--attacked', action='store_true', default=False,
                    help='Whether using the attacked graph')
parser.add_argument('--load_best', action='store_true', default=False, 
    help='Load the best model')
parser.add_argument('--save_file', type=str, default="",
                    help='Save file Path')
parser.add_argument('--ch_dir', type=str, default="",
                    help='Save file Path')
parser.add_argument('--dgi', action='store_true', default=False,
                    help='Whether using the dgi features')
parser.add_argument('--random_split', action='store_true', default=False,
                    help='Whether using random split')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if len(args.ch_dir)>0:
    import os
    os.chdir(args.ch_dir)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print('modified = ', args.modified)
# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, args.modified, args.attacked)
if args.dgi:
    dgi_features =  np.load('../LAGCN/data/' + args.dataset + '_dgi.npy')[0]
    dgi_features = torch.from_numpy(dgi_features).float()
    features = torch.cat([features, dgi_features],dim=-1)

if args.random_split:
    idx_val = list(idx_val.data.numpy())
    idx_test = list(idx_test.data.numpy())
    idx_total = idx_val+idx_test

    random.shuffle(idx_total)

    idx_val = torch.LongTensor(idx_total[:500])
    idx_test = torch.LongTensor(idx_total[500:])
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
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
    return acc_val


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test

model_file = 'model_save/' + args.dataset + '.pkl'
# Train model
t_total = time.time()
max_acc = 0
acc_list = []
for epoch in range(args.epochs):
    val_acc = train(epoch)
    if val_acc > max_acc:
        max_acc = val_acc
        torch.save(model.state_dict(), model_file)
        acc_list.append(val_acc)

if args.load_best:
    model.load_state_dict(torch.load(model_file))
print(max(acc_list))
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
acc_test = test()
if len(args.save_file)>0:
    with open(args.save_file,'a') as f:
        f.write('GCN %.4f'%acc_test)
        f.write('\n')
