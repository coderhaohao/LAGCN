import os
import time
import random
import torch
import argparse
import numpy as np
from preload import load_data
from torch.autograd import Variable
from model import *
# from edge import *
from utils import *
from sampler import *
from sklearn.metrics import accuracy_score
from preload import load_data, load_reddit_data
from ipdb import launch_ipdb_on_exception


def batch_eva(net, x, y, batch_size):
    out_list = []
    for idx in range(0, len(x),batch_size):
        end_idx = min(idx+batch_size,len(x))
        batch_nodes = x[idx:end_idx]
        output = net.forward(batch_nodes).data.cpu().numpy().argmax(axis=1)
        out_list.append(output)
    pred = np.concatenate(out_list, axis=0)
    acc = accuracy_score(y,pred)
    return acc

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="reddit",
                    help='Dataset to use.')
parser.add_argument('--batch_size', type=int, default=512,
                    help='Batch Size.')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='Hidden layer size')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--num_neigh0', type=int, default=25,
                    help='1-hop neighbor num')
parser.add_argument('--num_neigh1', type=int, default=10,
                    help='2-hop neighbor num')
parser.add_argument('--patience', type=int, default=20,
                    help='Patience number')
parser.add_argument('--evaluation_batch', type=int, default=20,
                    help='Batch per evaluation')
parser.add_argument('--evaluation_size', type=int, default=1000,
                    help='Batch per evaluation')
parser.add_argument('--modified', action='store_true', default=False,
                    help='Whether using the modified graph')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
with launch_ipdb_on_exception():
    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    model_name = 'model_save/' + args.dataset + '.pkl'

    adj, features, labels, train_index, val_index, test_index = load_reddit_data(args.cuda, args.modified)

    s = SamplerReddit(adj)
    sampler = s.normal_sample
    num_nodes = features.shape[0]
    num_feature = features.shape[1]
    num_classes = labels.shape[1]
    feat = torch.nn.Embedding(num_nodes, num_feature)
    feat.weight = torch.nn.Parameter(torch.FloatTensor(features), requires_grad=False)

    agg1 = MeanAggregator(feat, sampler, is_softgate=True, cuda=args.cuda)
    enc1 = Encoder(feat, num_feature, args.hidden_size, agg1, gcn=False, cuda=args.cuda)
    agg2 = MeanAggregator(
        lambda nodes: enc1(nodes).t(), sampler, is_softgate=True, cuda=args.cuda)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, args.hidden_size, agg2,
        base_model=enc1, gcn=False, cuda=args.cuda)
    enc1.num_sample = args.num_neigh1
    enc2.num_sample = args.num_neigh0
    graphsage = SupervisedGraphSage(num_classes, enc2, drop_out=0.5, cuda=args.cuda)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=1.0)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, graphsage.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    score_list = []
    times = []
    patience_count=0
    max_score=0
    start_time = time.time()
    for batch in range(100000):
        batch_nodes = random.sample(list(train_index), args.batch_size)
        optimizer.zero_grad()
        label = Variable(torch.LongTensor(labels[batch_nodes].argmax(axis=1)))
        if args.cuda:
            label = label.cuda()
        tr_output, loss = graphsage.loss(
            batch_nodes,
            label,
            is_train=True
        )
        loss.backward()
        optimizer.step()

        if batch%args.evaluation_batch ==0:
            va_nodes = random.sample(list(val_index), args.evaluation_size)
            tr_acc = batch_eva(graphsage,batch_nodes,labels[batch_nodes].argmax(axis=1), args.batch_size)
            va_acc = batch_eva(graphsage,va_nodes,labels[va_nodes].argmax(axis=1), 2*args.batch_size)
            if va_acc > max_score:
                max_score = va_acc
                patience_count = 0
                torch.save(graphsage.state_dict(), model_name)
            else:
                patience_count +=1
                if patience_count== args.patience:
                    print('Early Stop')
                    break
            end_time = time.time()
            time_cost = (end_time - start_time)/60
            times.append(end_time - start_time)
            start_time=end_time
            print('Epoch: %d,' %(batch),
                  '|Pat: %d/%d' % (patience_count, args.patience),
                  '|loss: %.4f' % loss.data.cpu().numpy(),
                  '|tr_acc: %.4f' % tr_acc,
                  '|va_acc: %.4f' % va_acc,
                  'cost %.2f m' % time_cost
                  )
    print('*'*20, 'Final Test', '*'*20)
    graphsage.load_state_dict(torch.load(model_name))
    ts_acc = batch_eva(graphsage,test_index,labels[test_index].argmax(axis=1), 2*args.batch_size)
    print('Test Acc:%.4f'%ts_acc)