import random
import numpy as np
import scipy.sparse as sp

def one_hot(values):
    n_values = np.max(values) + 1
    one_hot = np.eye(n_values)[values]
    return one_hot


def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to normal representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def att_del_edge(s, idx, lb_oh, att_num=1):
    neigh = list(s.neigh_map[idx])
    target_list = list(set(neigh).difference(set([idx])))
    edge_list = list(zip([idx] * len(target_list), target_list))
    gt = np.array([int(np.sum(lb_oh[a]*lb_oh[b])) for (a,b) in edge_list])
    same_idxs = list(np.where(gt>0)[0])
    if len(same_idxs) > 0:
        att_num = min(len(same_idxs), att_num)
        del_idx = random.sample(same_idxs, att_num)
        edge = [edge_list[_] for _ in del_idx]
    else:
        edge = []
    return edge


def att_del_process(s, adj, att_num, idx_list, lb_oh):
    if att_num==0:
        return adj
    edge_list = []
    for node in idx_list:
        del_edge = att_del_edge(s, node, lb_oh, att_num)
        edge_list.extend(del_edge)

    edge_array = np.array(edge_list)
    print(len(edge_array))
    adj_filt = adj.copy()
    adj_filt[edge_array[:, 0], edge_array[:, 1]] = 0
    adj_filt[edge_array[:, 1], edge_array[:, 0]] = 0
    return adj_filt
