import numpy as np
from utils import load_data, la_evulate, sgc_precompute
from preprocess import BatchTrain, gen_train_set, gen_train_reddit
from ipdb import launch_ipdb_on_exception
import torch
import torch.nn.modules
from model import Net
import argparse
import torch.utils.data as Data
from ipdb import launch_ipdb_on_exception
import time

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    help='Name of dataset.')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--method', type=str, default='dgi',
                    help='Type of embeddings')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--patience', type=int, default=30,
                    help='Patience batch of training.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--drop_rate', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--balance_rate', type=float, default=2,
                    help='Balance rate. (If set a negative number then take as defalut setting)')
parser.add_argument('--extra_rate', type=float, default=1,
                    help='Extra rate. (If set a negative number then take as defalut setting)')
# parser.add_argument('--test', action='store_true', default=False,
#                     help='Testing mode')

args, _ = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda:0" if args.cuda else "cpu")
model_name = args.dataset + '_' + args.method + '.pkl'
save_file = 'model_save/' + model_name

adj, features, lb_dict, tr_set, va_set, ts_set = load_data(args.dataset)

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

with launch_ipdb_on_exception():
#     if args.dataset=='reddit':
#         tr_edges, va_edges, ts_edges = gen_train_reddit(adj, lb_dict, tr_set, va_set, ts_set, args.balance_rate, 0)
#     else:
    tr_edges, va_edges, ts_edges = gen_train_reddit(adj, lb_dict, tr_set, va_set, ts_set, args.balance_rate, args.extra_rate,1)

    tr_data = tr_edges[:,:-1]
    tr_label = tr_edges[:,-1]
    va_data = va_edges[:, :-1]
    va_label = va_edges[:, -1]
    ts_data = ts_edges[:, :-1]
    ts_label = ts_edges[:, -1]

    X_train = torch.from_numpy(tr_data)
    X_val = torch.from_numpy(va_data)
    X_test = torch.from_numpy(ts_data)
    Y_train = torch.from_numpy(tr_label).long()
    Y_val = torch.from_numpy(va_label).long()
    Y_test = torch.from_numpy(ts_label).long()

    BATCH_SIZE = 256
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)

    def evaluate(net, X, Y, if_print=False):
        prediction = torch.max(net(X.to(args.device)), 1)[1]
        pred_y = prediction.cpu().data.numpy()
        target_y = Y.data.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        score = la_evulate(target_y, pred_y, if_print)
        return accuracy, score

    print(emb_size)
    net = Net(ne_array, word_num=word_num, emb_size=emb_size, trans_size=128,
              n_hidden=256, n_hidden1=256, n_output=2, drop_rate=0.5, device=args.device)  # define the network
    print(net)  # net architecture
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
    max_score_list = [0, 0, 0, 0, 0, 0]
    score_list = []
    batch = 0
    patience_count = 0
    patience_flag = False
    t0 = time.time()
    for t in range(100):
        if patience_flag:
            print('Early stop!!!')
            break
        for step, (b_x, b_y) in enumerate(train_loader):
            batch += 1
            net.train()
            out = net(b_x.to(args.device))  # input x and predict based on x
            loss = loss_func(out, b_y.to(args.device))  # must be (1. nn output, 2. target), the target label is NOT one-hotted

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            score_vle = []
            if step % 100 == 0:
                net.train()
                tr_acc, tr_score = evaluate(net, X_train, Y_train)
                score_vle.extend([tr_score[0], tr_score[1]])

                net.eval()
                va_acc, va_score = evaluate(net, X_val, Y_val, True)
                score_vle.extend([va_score[0], va_score[1]])
                ts_acc, ts_score = evaluate(net, X_test, Y_test, True)
                score_vle.extend([ts_score[0], ts_score[1]])
                score_list.append(score_vle)
                if score_vle[2] > max_score_list[2]:
                    max_score_list[2] = score_vle[2]
                    max_score_list[3] = score_vle[3]
                    patience_count = 0
                    torch.save(net.state_dict(), 'model_save/' + model_name)
                else:
                    patience_count += 1
                    if patience_count == args.patience:
                        patience_flag = True
                        break
                print('Epoch: %d(%d),' %(t, step),
                      '|Pat: %d/%d' % (patience_count, args.patience),
                      '|loss: %.4f' % loss.cpu().data.numpy(),
                      '|ts_acc: %.4f' % ts_acc)
    # net = Net(ne_array, word_num=word_num, emb_size=emb_size, trans_size=128,
              # n_hidden=256, n_hidden1=256, n_output=2, drop_rate=0.5)  # define the network
    t1 = time.time()
    print("Consuming %.2f second"%(t1-t0))
    net.load_state_dict(torch.load('model_save/' + model_name))
    net.eval()
    print('*'*20, 'Final Test', '*'*20)
    va_acc, va_score = evaluate(net, X_val, Y_val, True)
    ts_acc, ts_score = evaluate(net, X_test, Y_test, True)







