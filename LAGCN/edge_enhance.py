import time
import torch
import argparse
from datetime import datetime as dt
import random
import torch.nn as nn
import numpy as np
from model import Net
import scipy.sparse as sp
import torch.nn.functional as F
from utils import load_data, sgc_precompute
import argparse
from preprocess import BatchTrain
from ipdb import launch_ipdb_on_exception


def filter_neigh(neigh_map, idx, net, thre, lb_dict, is_cuda=False):
    neigh = list(neigh_map[idx])
    target_list = list(set(neigh).difference(set([idx])))
    edge_list = np.array(list(zip([idx] * len(target_list), target_list)))
    if len(edge_list) == 0:
        return []
    edge_tensor = torch.LongTensor(edge_list)
    if is_cuda:
        edge_tensor = edge_tensor.cuda()
    pred_edges_list = F.softmax(net(edge_tensor), dim=1)[:, 0].cpu().data.numpy()
    same_idx = list(np.where(pred_edges_list > thre)[0])
    # gt = np.array([1-int(np.sum(lb_dict[a]*lb_dict[b])) for (a,b) in edge_list])
    # neg_num = np.sum(gt)
    # acc_num = np.sum(gt[same_idx])
    pred_num = len(same_idx)
    tot_num = len(edge_list)
#     print(node, pred_num,acc_num)
#     return idx,np.array(edge_list)[same_idx], acc_num, neg_num,pred_num,tot_num
    return idx, np.array(edge_list)[same_idx]


def add_neigh(neigh_map, idx, net, total_neigh_num, thre, lb_dict, is_cuda=False):
#     print(idx)
    neigh = list(neigh_map[idx])
    if len(neigh)==0:
        return []
    sec_neigh = set()
    for node in neigh:
        if node in neigh_map:
            sec_neigh.update(neigh_map[node])
    target_list = list(sec_neigh.difference(set(neigh)))
    if len(target_list)==0:
        return []
    edge_list = np.array(list(zip([idx] * len(target_list), target_list)))
    edge_tensor = torch.LongTensor(edge_list)
    if is_cuda:
        edge_tensor = edge_tensor.cuda()
    # pred_edges_list = torch.max(net(edge_tensor), 1)[1].cpu().data.numpy()

    pred_edges_list = F.softmax(net(edge_tensor), dim=1)[:,1].cpu().data.numpy()
    same_idx = list(np.where(pred_edges_list > thre)[0])
    #
    edge_dict = dict(zip(np.array(target_list)[same_idx],pred_edges_list[same_idx]))

    # same_idx = list(np.where(pred_edges_list == 1)[0])

    need_num = total_neigh_num-len(neigh)
    # gt = np.array([int(np.sum(lb_dict[a]*lb_dict[b])) for (a, b) in edge_list])
    # acc_num = np.sum(pred_edges_list*gt)
#     print(same_idx)
    if need_num <= 0:
        return []
    else:
        added_list = sorted(edge_dict.keys(), key=lambda k: edge_dict[k], reverse=True)[:need_num]
        added_edge_list = np.array(list(zip([idx] * len(added_list), added_list)))
#         same_idx = random.sample(same_idx,need_num)
    # tot_num = np.sum(gt[same_idx])
    # return idx,np.array(edge_list)[same_idx], acc_num, tot_num,len(same_idx)
    return idx,added_edge_list


def edge_review(edge_array, lb_dict, is_print=True):
    edge_label = []
    for i in range(edge_array.shape[0]):
        edge = edge_array[i,:]
        try:
            edge_label.append(np.sum(lb_dict[edge[0]]*lb_dict[edge[1]]))
        except:
            print(edge)
    edge_lb = np.array(edge_label)
    pos_idx = np.where(edge_lb>0)[0]
    neg_idx = np.where(edge_lb==0)[0]
    if is_print:
        print('pos_edge:{}, neg_edge:{}'.format(len(pos_idx), len(neg_idx)))
    return len(pos_idx), len(neg_idx)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    help='Name of dataset.')
parser.add_argument('--method', type=str, default='mul',
                    help='Type of embeddings')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--patience', type=int, default=30,
                    help='Patience batch of training.')
parser.add_argument('--no_add', action='store_true', default=False,
                    help='No adder.')
parser.add_argument('--no_filter', action='store_true', default=False,
                    help='No filter.')

args = parser.parse_args()
print('\n'.join([(str(_)+':'+str(vars(args)[_])) for _ in vars(args).keys()]))
dataset = args.dataset

np.random.seed(args.seed)
torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

method = args.method

thre = 0.7
if dataset in ['cora', 'citeseer']:
    total_neigh = 6
if dataset in ['pubmed']:
    total_neigh = 30
if_add = not args.no_add
if_filt = not args.no_filter
adj, features, lb_dict, tr_set, va_set, ts_set = load_data(dataset)
node_set = list(tr_set) + list(va_set) + list(ts_set)

save_file = 'model_save/' + dataset + '_' + method + '.pkl'
if args.method == 'raw':
    ne_array = features
elif args.method == 'dgi':
    ne_array = np.load('data/' + args.dataset + '_' + args.method + '.npy')[0]
elif args.method == 'mul':
    ne_array = sgc_precompute(features, adj, 2)
else:
    raise ValueError('Embedding method undefined:%s' % args.method)
if not isinstance(ne_array, np.ndarray):
    ne_array = ne_array.toarray()

word_num = ne_array.shape[0]
emb_size = ne_array.shape[1]


start_time = dt.now()
print("start to load model and generate neigh_map")
net = Net(ne_array, word_num=word_num, emb_size=emb_size, trans_size=128,
          n_hidden=256, n_hidden1=256, n_output=2, drop_rate=0.5, device='cpu')  # define the network
print(net)  # net architecture
net.load_state_dict(torch.load(save_file))
net.eval()
neigh_map={}
for node_0, node_1 in zip(adj.nonzero()[0],adj.nonzero()[1]):
    if node_0 not in neigh_map:
        neigh_map[node_0] = [node_1]
    else:
        neigh_map[node_0].append(node_1)
# neigh_map = dict([(_, list(np.nonzero(sp.lil_matrix(adj[_,:]))[1])) for _ in range(adj.shape[0])])
for _ in range(adj.shape[0]):
    if _ not in neigh_map:
        neigh_map[_] = []

end_time = dt.now()
print("cost time:", end_time - start_time)
start_time = end_time
print("start to add neighs")

if if_add:
    res_list = []
    edge_list = []
    for node in node_set:
        new_neigh = add_neigh(neigh_map, node, net, total_neigh, thre, lb_dict, is_cuda=False)
        res_list.append(new_neigh)
    for r in res_list:
        if len(r) > 0 and r[1].shape[0] > 0:
            edge_list.append(r[1])
    # print("the number of add edges:", len(edge_list))
    add_edge_array = np.vstack(edge_list)
    # print("the number of add edges:", len(add_edge_array))
    edge_review(add_edge_array, lb_dict)
    adj[add_edge_array[:, 0], add_edge_array[:, 1]] = 1

end_time = dt.now()
print("cost time:", end_time - start_time)
start_time = end_time
print("start to filt neighs")

if if_filt:
    res_list = []
    edge_list = []
    for node in node_set:
        res_list.append(filter_neigh(neigh_map, node, net, thre, lb_dict, is_cuda=False))
    for r in res_list:
        if len(r) > 0 and r[1].shape[0] > 0:
            edge_list.append(r[1])
    # print("the number of filt edges:", len(edge_list))
    filt_edge_array = np.vstack(edge_list)
    edge_review(filt_edge_array, lb_dict)
    adj[filt_edge_array[:, 0], filt_edge_array[:, 1]] = 0

end_time = dt.now()
print("cost time:", end_time - start_time)
start_time = end_time
print("start to generate new neighs")

new_neigh_map = neigh_map = dict([(_, list(np.nonzero(sp.lil_matrix(adj[_, :]))[1])) for _ in range(adj.shape[0])])

end_time = dt.now()
print("cost time:", end_time - start_time)
start_time = end_time
print("start to generate write files")

import pickle as pkl

with open('data/ind.' + dataset + '.graph_lite', 'wb') as f:
    pkl.dump(new_neigh_map, f)

end_time = dt.now()
print("cost time:", end_time - start_time)
start_time = end_time
print("finish !!")
