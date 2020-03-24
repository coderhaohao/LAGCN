import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
from scipy.sparse.linalg import norm as sparsenorm
from scipy.linalg import qr
import json
from networkx.readwrite import json_graph


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Load data."""
    path = '../LAGCN/'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph_lite']
    objects = []
    for i in range(len(names)):
        with open(path+"data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(path+"data/ind.{}.test.index".format(dataset_str))
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

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def loadRedditFromNPZ(dataset_dir, modified):
    if modified:
        adj = sp.load_npz(dataset_dir+"reddit_adj_lite.npz")
    else:
        adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(cuda=True, modified=False):
    path = '../LAGCN/'
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(path+"data/", modified)
    labels = np.zeros(adj.shape[0])
    labels[train_index] = y_train
    labels[val_index] = y_val
    labels[test_index] = y_test
    labels = one_hot(labels.astype(np.int))
    if not modified:
        adj = adj + adj.T + sp.eye(adj.shape[0])
    features = np.array(features)
    # features = (features-features.mean(dim=0))/features.std(dim=0)

    return adj, features, labels, train_index, val_index, test_index


def one_hot(values):
    n_values = np.max(values) + 1
    one_hot = np.eye(n_values)[values]
    return one_hot


