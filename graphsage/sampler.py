import numpy as np
import random
import scipy.sparse as sp
import time
from itertools import chain

import multiprocessing as mul
ADJ, ADJ_NORM = 1,1

def normalize_adj(adj):
    """Row-normalize feature matrix and convert to normal representation"""
    rowsum = np.array(np.sum(adj, axis=1))
    r_inv = (1 / rowsum).reshape([-1])
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    return adj


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(alias_tuple):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    J, q = alias_tuple
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

def com_adj_dict(idx):
    global ADJ, ADJ_NORM
    neigh_list = np.nonzero(ADJ[idx])[1]
    prob_list = np.array(ADJ_NORM[idx, neigh_list].todense())[0]
    softmax_list = np.array(ADJ[idx, neigh_list].todense())[0]
    softmax_dict = dict(zip(neigh_list, softmax_list))
    J_vle, q_vle = alias_setup(prob_list)
    return neigh_list, (J_vle, q_vle), softmax_dict

def com_adj_dict_reddit(idx):
    global ADJ, ADJ_NORM
    neigh_list = np.nonzero(ADJ[idx])[1]
    return neigh_list


class Sampler():
    def __init__(self, adj, loop=True, n_jobs=0):
        # if ~loop:
        #     N = adj.shape[0]
        #     adj[list(range(N)), list(range(N))] = 0
        neigh_map, alias_map, softmax_map = self.compute(adj, n_jobs)
        self.neigh_map = neigh_map
        self.alias_map = alias_map
        self.softmax_map = softmax_map

    def reset(self, adj, loop=True, n_jobs=0):
        if ~loop:
            N = adj.shape[0]
            adj[list(range(N)), list(range(N))] = 0
        neigh_map, alias_map, softmax_map = self.compute(adj, n_jobs)
        self.neigh_map = neigh_map
        self.alias_map = alias_map
        self.softmax_map = softmax_map

    def update(self, node_list, neighs_list):
        if type(node_list) != type([]):
            node_list = [node_list]
        if type(neighs_list[0]) != type([]):
            neighs_list = [neighs_list]
        for i in range(len(node_list)):
            node = node_list[i]
            neigh = neighs_list[i]
            self.neigh_map[node] = neigh

    def compute(self, adj, n_jobs=0):
        time0 = time.time()
        neigh_map = {}  # map node to Neighbour list
        alias_map = {}  # map node to alias tuple
        softmax_map = {}  # map node to softmax edge prediction vle
        adj_norm = normalize_adj(adj)
        adj = sp.lil_matrix(adj)
        adj_norm = sp.lil_matrix(adj_norm)
        if n_jobs == 0:
            for i in range(adj.shape[0]):
                neigh_list = np.nonzero(adj[i])[1]
                prob_list = np.array(adj_norm[i, neigh_list].todense())[0]
                softmax_list = np.array(adj[i, neigh_list].todense())[0]
                softmax_dict = dict(zip(neigh_list, softmax_list))
                J_vle, q_vle = alias_setup(prob_list)
                neigh_map[i] = neigh_list
                alias_map[i] = (J_vle, q_vle)
                softmax_map[i] = softmax_dict
        else:
            global ADJ, ADJ_NORM
            ADJ = adj
            ADJ_NORM = adj_norm
            with mul.Pool(n_jobs) as pool:
                res_list = pool.map(com_adj_dict, list(range(adj.shape[0])))
            for i, res in enumerate(res_list):
                neigh_map[i] = res[0]
                alias_map[i] = res[1]
                softmax_map[i] = res[2]
            del ADJ
            del ADJ_NORM
        time1 = time.time()
        print('Compute sampler in %.2f min' % ((time1 - time0) / 60))
        return neigh_map, alias_map, softmax_map

    def alias_sample(self, nodes, num_sample):
        if type(nodes) != type([]):
            nodes = [nodes]
        samp_list = []
        weight_list = []
        for n in nodes:
            n_samps = []
            n_neigh = self.neigh_map[n]
            n_alias = self.alias_map[n]
            n_softmax = self.softmax_map[n]
            for i in range(num_sample):
                n_samps.append(
                    n_neigh[
                        alias_draw(n_alias)])
            n_samps = list(set(n_samps))
            n_soft_vle = [n_softmax[x] for x in n_samps]
            samp_list.append(set(n_samps))
            weight_dict = dict(zip(n_samps, n_soft_vle))
            weight_list.append(weight_dict)
        return samp_list, weight_list

    def normal_sample(self, nodes, num_sample):
        samp_list = []
        weight_list = []
        for n in nodes:
            n_neigh = self.neigh_map[n]
            if not num_sample is None:
                n_samps = set(
                    random.sample(list(n_neigh), num_sample)
                ) if len(n_neigh) >= num_sample else set(n_neigh)
            else:
                n_samps = set(n_neigh)
            samp_list.append(n_samps)
            n_soft_vle = [1] * len(n_samps)
            weight_dict = dict(zip(n_samps, n_soft_vle))
            weight_list.append(weight_dict)
        return samp_list, weight_list

    def ml_sample(self, node_list, level, samp_num=None, is_distinct=False, is_filt_self=False):
        step_list = []
        step_list.append([[_] for _ in node_list])
        for i in range(level):
            last_list = step_list[-1]
            if is_distinct:
                step_list.append([list(set(chain(*[list(self.neigh_map[_]) for _ in nei]))) for nei in last_list])
            else:
                step_list.append([list(chain(*[list(self.neigh_map[_]) for _ in nei])) for nei in last_list])
        if is_filt_self:
            for key in range(0, level):
                for i in range(key + 1, level + 1, 1):
                    step_list[i] = [
                        list(filter(
                            lambda x: x not in set(step_list[key][j]), step_list[i][j]))
                        for j in range(len(step_list[key]))]
        else:
            for key in range(0, level):
                for i in range(key + 1, level + 1, 1):
                    step_list[i] = [
                        list(set(filter(
                            lambda x: x not in set(step_list[key][j]), step_list[i][j])))
                        for j in range(len(step_list[key]))]

        if samp_num:
            for i in range(level):
                nei_list = step_list[i + 1]
                s_num = samp_num[i]
                step_list[i + 1] = [random.sample(nei, s_num) if len(nei) > s_num else nei for nei in nei_list]

        total_list = list(set(chain(*[list(chain(*step)) for step in step_list])))
        node_map = dict(zip(total_list, range(len(total_list))))

        idx_list = []
        for step_ind, step in enumerate(step_list):
            row_idx = list(chain(*[[i for _ in nei] for i, nei in enumerate(step)]))
            col_idx = list(chain(*[[node_map[_] for _ in nei] for nei in step]))
            idx_list.append((row_idx, col_idx))
            #         return step_list
        return total_list, idx_list, len(node_list)


class SamplerReddit():
    def __init__(self, adj, n_jobs=0):
        neigh_map = self.compute(adj, n_jobs)
        self.neigh_map = neigh_map

    def compute(self, adj, n_jobs=0):
        time0 = time.time()
        neigh_map = {}  # map node to Neighbour list
        for i in range(adj.shape[0]):
            neigh_list = np.nonzero(adj[i])[1]
            neigh_map[i] = set(list(neigh_list))
            if i%10000==0:
                print('%d nodes completed.'%i)
        time1 = time.time()
        print('Compute sampler in %.2f min' % ((time1 - time0) / 60))
        return neigh_map


    def normal_sample(self, nodes, num_sample):
        samp_list = []
        weight_list = []
        for n in nodes:
            n_neigh = self.neigh_map[n]
            if not num_sample is None:
                n_samps = set(
                    random.sample(list(n_neigh), num_sample)
                ) if len(n_neigh) >= num_sample else set(n_neigh)
            else:
                n_samps = set(n_neigh)
            samp_list.append(n_samps)
            n_soft_vle = [1] * len(n_samps)
            weight_dict = dict(zip(n_samps, n_soft_vle))
            weight_list.append(weight_dict)
        return samp_list, weight_list
