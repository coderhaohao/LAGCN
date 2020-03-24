import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter

# Arguments
args = get_citation_args()

if len(args.ch_dir)>0:
    import os
    os.chdir(args.ch_dir)

import os
print(os.getcwd())

if not args.non_tuned:
    if args.model == "SGC":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# setting random seeds
set_seed(args.seed, args.cuda)

adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda, args.modified, args.attacked)

model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj, args.degree)
print("{:.4f}s".format(precompute_time))
model_file = 'model_save/' + args.dataset + '.pkl'


def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    max_acc = 0
    acc_list = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)
        print(acc_val)
        if acc_val > max_acc:
            max_acc = acc_val
            torch.save(model.state_dict(), model_file)
            acc_list.append(acc_val)
    train_time = perf_counter()-t

    with torch.no_grad():
        if args.load_best:
            model.load_state_dict(torch.load(model_file))
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time

def test_regression(model, test_features, test_labels):
    model.eval()
    if args.load_best:
        model.load_state_dict(torch.load(model_file))
    return accuracy(model(test_features), test_labels)

if args.model == "SGC":
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                     args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])

if len(args.save_file)>0:
    with open(args.save_file,'a') as f:
        f.write('SGC %.4f'%acc_test)
        f.write('\n')
print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
