import time
import numpy as np
import random
import pandas as pd

class BatchTrain:
    def __init__(
            self, adj, lb_dict, tr_set=None,
            va_set=None, ts_set=None,
            balance_rate=0, extra_rate=0):
        tr_set = set(tr_set)
        va_set = set(va_set).union(tr_set)
        ts_set = set(ts_set).union(va_set)
        tr_edge_list = []
        va_edge_list = []
        ts_edge_list = []

        for a in range(adj.shape[0]):
            neigh_list = np.nonzero(adj[a, :])[1]
            for b in neigh_list:
                if a in ts_set and b in ts_set:
                    edge_lb = 1 if np.dot(lb_dict[a], lb_dict[b]) > 0 else 0
                    edge_tuple = (a, b, edge_lb)
                    if a in tr_set and b in tr_set:
                        tr_edge_list.append(edge_tuple)
                        continue
                    if a in va_set and b in va_set:
                        va_edge_list.append(edge_tuple)
                        continue
                    if a in ts_set and b in ts_set:
                        ts_edge_list.append(edge_tuple)
                        continue
                else:
                    print("{},{} not in above set".format(a, b))

        self.tr_edges = np.array(tr_edge_list)
        self.va_edges = np.array(va_edge_list)
        self.ts_edges = np.array(ts_edge_list)
        self.tr_pos = self.tr_edges[self.tr_edges[:, -1] == 1]
        self.tr_neg = self.tr_edges[self.tr_edges[:, -1] == 0]

        if balance_rate > 0:
            bal_edge_list = []
            pos_num = self.tr_pos.shape[0]
            neg_num = self.tr_neg.shape[0]
            for i in range(neg_num, int(balance_rate * pos_num)):
                a, b = random.sample(list(tr_set), 2)
                lb = int(np.dot(lb_dict[a], lb_dict[b]))
                while lb != 0:
                    a, b = random.sample(list(tr_set), 2)
                    lb = int(np.dot(lb_dict[a], lb_dict[b]))
                bal_edge_list.append((a, b, lb))
            for _ in bal_edge_list:
                print(_)
            tr_edge_list.extend(bal_edge_list)


        if extra_rate > 0:
            extra_num = int(0.5 * extra_rate * len(tr_edge_list))
            ext_pos_edge_list = []
            ext_neg_edge_list = []
            while len(ext_pos_edge_list) < extra_num or len(ext_neg_edge_list) < extra_num:
                a, b = random.sample(list(tr_set), 2)
                lb = int(np.dot(lb_dict[a], lb_dict[b]))
                if lb > 0:
                    if len(ext_pos_edge_list) < extra_num:
                        ext_pos_edge_list.append((a, b, lb))
                    continue
                else:
                    if len(ext_neg_edge_list) < extra_num:
                        ext_neg_edge_list.append((a, b, lb))
                    continue
            ext_edge_list = ext_pos_edge_list + ext_neg_edge_list
            tr_edge_list.extend(ext_edge_list)
        self.tr_edges = np.array(tr_edge_list)
        self.tr_pos = self.tr_edges[self.tr_edges[:, -1] == 1]
        self.tr_neg = self.tr_edges[self.tr_edges[:, -1] == 0]

        print('number of training edges:', len(self.tr_edges))
        print("number of validation edges:", len(self.va_edges))
        print("number of testing edges:", len(self.ts_edges))
        print("number of pos edge for training:", len(self.tr_pos))
        print("number of neg edge for training:", len(self.tr_neg))
        self.ts_pos = self.ts_edges[self.ts_edges[:, -1] == 1]
        print("number of pos edge for testing:", len(self.tr_pos))
        self.ts_neg = self.ts_edges[self.ts_edges[:, -1] == 0]
        print("number of neg edge for testing:", len(self.tr_neg))


    def gen_batch(self, batch_size):
        if batch_size % 2 != 0:
            raise ValueError('Batch Size should be a even number.')
        half_batch = int(batch_size / 2)
        rand_list = random.sample(list(range(self.tr_pos.shape[0])), half_batch)
        pos_array = self.tr_pos[rand_list, :]
        rand_list = random.sample(list(range(self.tr_neg.shape[0])), half_batch)
        neg_array = self.tr_neg[rand_list, :]
        batch_array = np.concatenate([pos_array, neg_array], axis=0)
        np.random.shuffle(batch_array)
        return batch_array


def gen_train_set_old(
        adj, lb_oh, tr_set=None,
        va_set=None, ts_set=None,
        balance_rate=0, extra_rate=0):
    tr_set = set(tr_set)
    va_set = set(va_set).union(tr_set)
    ts_set = set(ts_set).union(va_set)
    tr_edge_list = []
    tr_edlb_list = []
    va_edge_list = []
    va_edlb_list = []
    ts_edge_list = []
    ts_edlb_list = []

    for a in range(adj.shape[0]):
        neigh_list = np.nonzero(adj[a, :])[1]
        for b in neigh_list:
            if a in tr_set and b in tr_set:
                tr_edge_list.append((a, b))
                lb = int(np.dot(lb_oh[a], lb_oh[b]))
                tr_edlb_list.append(lb)
                continue
            if a in va_set and b in va_set:
                va_edge_list.append((a, b))
                lb = int(np.dot(lb_oh[a], lb_oh[b]))
                va_edlb_list.append(lb)
                continue
            if a in ts_set and b in ts_set:
                ts_edge_list.append((a, b))
                lb = int(np.dot(lb_oh[a], lb_oh[b]))
                ts_edlb_list.append(lb)
                continue

    # raw_tr_edge = tr_edge_list.copy()
    # raw_tr_edlb = tr_edlb_list.copy()

    if balance_rate > 0:
        count = pd.value_counts(tr_edlb_list)
        pos_num = count[1]
        neg_num = count[0]
        for i in range(neg_num, int(balance_rate * pos_num)):
            a, b = random.sample(list(tr_set), 2)
            lb = int(np.dot(lb_oh[a], lb_oh[b]))
            while lb != 0:
                a, b = random.sample(list(tr_set), 2)
                lb = int(np.dot(lb_oh[a], lb_oh[b]))
            tr_edge_list.append((a, b))
            tr_edlb_list.append(lb)

    if extra_rate > 0:
        extra_num = int(0.5 * extra_rate * len(tr_edge_list))
        pos_edge_list = []
        pos_edlb_list = []
        neg_edge_list = []
        neg_edlb_list = []
        while len(pos_edge_list) < extra_num or len(neg_edge_list) < extra_num:
            a, b = random.sample(list(tr_set), 2)
            lb = int(np.dot(lb_oh[a], lb_oh[b]))
            if lb > 0:
                if len(pos_edge_list) < extra_num:
                    pos_edge_list.append((a, b))
                    pos_edlb_list.append(lb)
                continue
            else:
                if len(neg_edge_list) < extra_num:
                    neg_edge_list.append((a, b))
                    neg_edlb_list.append(lb)
                continue
        tr_edge_list.extend(pos_edge_list)
        tr_edge_list.extend(neg_edge_list)
        tr_edlb_list.extend(pos_edlb_list)
        tr_edlb_list.extend(neg_edlb_list)

    tr_edlb_list = np.array(tr_edlb_list)
    va_edlb_list = np.array(va_edlb_list)
    ts_edlb_list = np.array(ts_edlb_list)
    tr_edges = [(tr_edge_list[_][0], tr_edge_list[_][1], tr_edlb_list[_]) for _ in range(len(tr_edge_list))]
    va_edges = [(va_edge_list[_][0], va_edge_list[_][1], va_edlb_list[_]) for _ in range(len(va_edge_list))]
    ts_edges = [(ts_edge_list[_][0], ts_edge_list[_][1], ts_edlb_list[_]) for _ in range(len(ts_edge_list))]

    tr_edges = np.array(tr_edges)
    va_edges = np.array(va_edges)
    ts_edges = np.array(ts_edges)

    print('number of training edges:', len(tr_edges))
    print("number of validation edges:", len(va_edges))
    print("number of testing edges:", len(ts_edges))
    tr_pos = tr_edges[tr_edges[:, -1] == 1]
    print("number of pos edge for training:", len(tr_pos))
    tr_neg = tr_edges[tr_edges[:, -1] == 0]
    print("number of neg edge for training:", len(tr_neg))
    ts_pos = ts_edges[ts_edges[:, -1] == 1]
    print("number of pos edge for testing:", len(ts_pos))
    ts_neg = ts_edges[ts_edges[:, -1] == 0]
    print("number of neg edge for testing:", len(ts_neg))
    return tr_edges, va_edges, ts_edges



def gen_train_set(
        adj, lb_oh, tr_set=None,
        va_set=None, ts_set=None,
        balance_rate=0, extra_rate=0, test=False):
    tr_set = set(tr_set)
    va_set = set(va_set).union(tr_set)
    ts_set = set(ts_set).union(va_set)
    tr_edge_list = []
    tr_edlb_list = []
    va_edge_list = []
    va_edlb_list = []
    ts_edge_list = []
    ts_edlb_list = []
    time0 = time.time()

    iter_num = adj.shape[0]
    if test:
        iter_num= min(30000,iter_num)
    for a in range(iter_num):
        neigh_list = np.nonzero(adj[a, :])[1]
        if a % 10000 == 0:
            print('%d nodes finished!!!' %a)
        for b in neigh_list:
            if a in tr_set and b in tr_set:
                tr_edge_list.append((a, b))
                lb = 1 if np.dot(lb_oh[a], lb_oh[b]) > 0 else 0
                tr_edlb_list.append(lb)
                continue
            if a in va_set and b in va_set:
                va_edge_list.append((a, b))
                lb = 1 if np.dot(lb_oh[a], lb_oh[b]) > 0 else 0
                va_edlb_list.append(lb)
                continue
            if a in ts_set and b in ts_set:
                ts_edge_list.append((a, b))
                lb = 1 if np.dot(lb_oh[a], lb_oh[b]) > 0 else 0
                ts_edlb_list.append(lb)
                continue
    time1 = time.time()

    pos_num = np.sum(tr_edlb_list)
    neg_num = len(tr_edlb_list) - pos_num
    print('Raw pos edges:%d, neg edges:%d'%(pos_num, neg_num))
    if balance_rate > 0:
        for i in range(neg_num, int(balance_rate * pos_num)):
            if i %10000 ==0:
                print('%d/%d neg edges filled' %(i, int(balance_rate * pos_num)))
            a, b = random.sample(list(tr_set), 2)
            lb = 1 if np.dot(lb_oh[a], lb_oh[b]) > 0 else 0
            while lb != 0:
                a, b = random.sample(list(tr_set), 2)
                lb = 1 if np.dot(lb_oh[a], lb_oh[b]) > 0 else 0
            tr_edge_list.append((a, b))
            tr_edlb_list.append(lb)

    time2 = time.time()

    if extra_rate > 0:
        extra_num = int(0.5 * extra_rate * len(tr_edge_list))
        pos_edge_list = []
        pos_edlb_list = []
        neg_edge_list = []
        neg_edlb_list = []
        while len(pos_edge_list) < extra_num or len(neg_edge_list) < extra_num:
            if (len(pos_edge_list) + len(neg_edge_list)) %10000==0:
                print('Extra filling... Pos:%d/%d, Neg:%d/%d'%(
                    len(pos_edge_list),extra_num,len(neg_edge_list),extra_num
                ))
            a, b = random.sample(list(tr_set), 2)
            lb = 1 if np.dot(lb_oh[a], lb_oh[b]) > 0 else 0
            if lb > 0:
                if len(pos_edge_list) < extra_num:
                    pos_edge_list.append((a, b))
                    pos_edlb_list.append(lb)
                continue
            else:
                if len(neg_edge_list) < extra_num:
                    neg_edge_list.append((a, b))
                    neg_edlb_list.append(lb)
                continue
        tr_edge_list.extend(pos_edge_list)
        tr_edge_list.extend(neg_edge_list)
        tr_edlb_list.extend(pos_edlb_list)
        tr_edlb_list.extend(neg_edlb_list)
    tr_edlb_list = np.array(tr_edlb_list)
    va_edlb_list = np.array(va_edlb_list)
    ts_edlb_list = np.array(ts_edlb_list)
    tr_edges = [(tr_edge_list[_][0], tr_edge_list[_][1], tr_edlb_list[_]) for _ in range(len(tr_edge_list))]
    va_edges = [(va_edge_list[_][0], va_edge_list[_][1], va_edlb_list[_]) for _ in range(len(va_edge_list))]
    ts_edges = [(ts_edge_list[_][0], ts_edge_list[_][1], ts_edlb_list[_]) for _ in range(len(ts_edge_list))]

    tr_edges = np.array(tr_edges)
    va_edges = np.array(va_edges)
    ts_edges = np.array(ts_edges)
    time3 = time.time()

    print('number of training edges:', len(tr_edges))
    print("number of validation edges:", len(va_edges))
    print("number of testing edges:", len(ts_edges))
    tr_pos = tr_edges[tr_edges[:, -1] == 1]
    print("number of pos edge for training:", len(tr_pos))
    tr_neg = tr_edges[tr_edges[:, -1] == 0]
    print("number of neg edge for training:", len(tr_neg))
    ts_pos = ts_edges[ts_edges[:, -1] == 1]
    print("number of pos edge for testing:", len(ts_pos))
    ts_neg = ts_edges[ts_edges[:, -1] == 0]
    print("number of neg edge for testing:", len(ts_neg))
    print('Raw Edge Compute in %.4f min' % ((time1 - time0) / 60))
    print('Balance Edge Compute in %.4f min' % ((time2 - time1) / 60))
    print('Extra Edge Compute in %.4f min' % ((time3 - time2) / 60))

    return tr_edges, va_edges, ts_edges