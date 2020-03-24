"""
Utils kits for LAGCN,
including:
1. data_loader
2. data_output
"""

import sys
import time
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from itertools import chain



def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_transductive(dataset_str, root_dir='', return_label=False, modified=False, attacked=False):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    if modified:
        names[-1] = 'graph_lite'
    if attacked:
        names[-1] = 'graph_attack'
    objects = []
    for i in range(len(names)):
        with open(root_dir+"data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:

            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally)-500)
    idx_val = range(len(ally)-500, len(ally))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    if return_label:
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,labels
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def loadRedditFromNPZ(dataset_dir, root_dir='', modified=False):
    if modified:
        adj = sp.load_npz(root_dir + 'data/' + dataset_dir + "_adj_lite.npz")
    else:
        adj = sp.load_npz(root_dir+'data/' + dataset_dir+"_adj.npz")
    data = np.load(root_dir+'data/' + dataset_dir+".npz")

    return (adj, data['feats'], data['y_train'], data['y_val'],
            data['y_test'], data['train_index'], data['val_index'],
            data['test_index'])


def load_data(dataset, modified=False, attacked=False):
    if dataset in ['cora', 'citeseer', 'pubmed']:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_transductive(
            dataset, modified=modified, attacked=attacked)
        tr_idx = list(np.where(train_mask)[0])
        va_idx = list(np.where(val_mask)[0])
        ts_idx = list(np.where(test_mask)[0])
        tr_lb = y_train[np.where(train_mask)[0]]
        va_lb = y_val[np.where(val_mask)[0]]
        ts_lb = y_test[np.where(test_mask)[0]]
        labels = list(chain(*[list(tr_lb), list(va_lb), list(ts_lb)]))
        idxs = list(chain(*[list(tr_idx), list(va_idx), list(ts_idx)]))
        lb_dict = dict(list(zip(idxs, labels)))
        tr_set = set(tr_idx)
        va_set = set(va_idx)
        ts_set = set(ts_idx)
    elif dataset in ['reddit']:
        adj, features, y_train, y_val, y_test, tr_idx, va_idx, ts_idx = loadRedditFromNPZ(dataset, modified=modified)
        if not modified:
            adj = adj + adj.T
        train_lb_list = list(zip(*[list(tr_idx), list(one_hot(y_train))]))
        val_lb_list = list(zip(*[list(va_idx), list(one_hot(y_val))]))
        test_lb_list = list(zip(*[list(ts_idx), list(one_hot(y_test))]))
        lb_list = train_lb_list + val_lb_list + test_lb_list
        lb_dict = dict(lb_list)
        tr_set = set(tr_idx)
        va_set = set(va_idx)
        ts_set = set(ts_idx)
    else:
        raise ValueError("dataset string is not allowable")
    return adj, features, lb_dict, tr_set, va_set, ts_set

def load_data_labels(dataset, modified=False):
    if dataset in ['cora', 'citeseer', 'pubmed']:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,labels = load_transductive(
            dataset, return_label=True, modified=modified)
        tr_idx = list(np.where(train_mask)[0])
        va_idx = list(np.where(val_mask)[0])
        ts_idx = list(np.where(test_mask)[0])
        tr_lb = y_train[np.where(train_mask)[0]]
        va_lb = y_val[np.where(val_mask)[0]]
        ts_lb = y_test[np.where(test_mask)[0]]
        # labels = list(chain(*[list(tr_lb), list(va_lb), list(ts_lb)]))
        # idxs = list(chain(*[list(tr_idx), list(va_idx), list(ts_idx)]))
        # lb_dict = dict(list(zip(idxs, labels)))
        tr_set = set(tr_idx)
        va_set = set(va_idx)
        ts_set = set(ts_idx)
    elif dataset in ['reddit']:
        adj, features, y_train, y_val, y_test, tr_idx, va_idx, ts_idx = loadRedditFromNPZ(dataset, modified=modified)
        if not modified:
            adj = adj + adj.T
        train_lb_list = list(zip(*[list(tr_idx), list(one_hot(y_train))]))
        val_lb_list = list(zip(*[list(va_idx), list(one_hot(y_val))]))
        test_lb_list = list(zip(*[list(ts_idx), list(one_hot(y_test))]))
        labels = np.zeros(adj.shape[0])
        labels[tr_idx] = y_train
        labels[va_idx] = y_val
        labels[ts_idx] = y_test
        labels = one_hot(labels.astype(np.int))
        lb_list = train_lb_list + val_lb_list + test_lb_list
        lb_dict = dict(lb_list)
        tr_set = set(tr_idx)
        va_set = set(va_idx)
        ts_set = set(ts_idx)
    else:
        raise ValueError("dataset string is not allowable")
    return adj, features, labels, tr_set, va_set, ts_set


def one_hot(values):
    n_values = np.max(values) + 1
    one_hot = np.eye(n_values)[values]
    return one_hot


def la_evulate(y_true, y_pred, if_print=False):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    y_auc = (y_pred == y_true)
    P = np.sum(y_pred)
    N = y_pred.shape[0] - P
    T = np.sum(y_true)
    F = y_true.shape[0] - T
    TP = np.sum(y_true[y_auc])
    FP = np.sum(y_pred[~y_auc])
    FN = np.sum((1-y_true)[y_auc])
    TN = np.sum(y_true[~y_auc])
    p_vle = TP/T
    q_vle = FP/F
    pre_vle = TP/P
    if if_print:
        print('p_vle: %.2f, q_vle:%.2f, p-q: %.2f, Pre: %.2f'%(p_vle, q_vle, p_vle-q_vle, pre_vle))
    return p_vle-q_vle, pre_vle


def sgc_precompute(features, adj, degree):
    t1 = time.time()
    for i in range(degree):
        features = adj*features
    t2 = time.time()
    print('Multiplying in %.2f Second' %(t2-t1))
    return features

