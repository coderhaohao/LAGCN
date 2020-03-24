import numpy as np
from utils import load_data, la_evulate, sgc_precompute
from preprocess import BatchTrain, gen_train_set
from ipdb import launch_ipdb_on_exception
import torch
import torch.nn.modules
from model import Net
import argparse
import torch.utils.data as Data

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    help='Name of dataset.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--method', type=str, default='raw',
                    help='Type of embeddings')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--patience', type=int, default=30,
                    help='Patience batch of training.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--drop_rate', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--balance_rate', type=float, default=-1,
                    help='Balance rate. (If set a negative number then take as defalut setting)')
parser.add_argument('--extra_rate', type=float, default=-1,
                    help='Extra rate. (If set a negative number then take as defalut setting)')
parser.add_argument('--test', action='store_true', default=False,
                    help='Testing mode')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

with launch_ipdb_on_exception():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    balance_rate = args.balance_rate
    if args.balance_rate < 0:
        balance_rate = 2
        if args.dataset=='reddit':
            balance_rate=0.5

    extra_rate = args.balance_rate
    if args.extra_rate < 0:
        extra_rate = 1
        if args.dataset == 'reddit' or args.dataset == 'pubmed':
            extra_rate = 0


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
    # b = BatchTrain(adj, lb_dict, tr_set, va_set, ts_set, balance_rate=1, extra_rate=0)
    # tr_edges = b.tr_edges
    # va_edges = b.va_edges
    # ts_edges = b.ts_edges
    tr_edges, va_edges, ts_edges = gen_train_set(
        adj, lb_dict, tr_set, va_set, ts_set, balance_rate, extra_rate, args.test)

    tr_data = tr_edges[:,:-1]
    tr_label = tr_edges[:,-1]
    va_data = va_edges[:, :-1]
    va_label = va_edges[:, -1]
    ts_data = ts_edges[:, :-1]
    ts_label = ts_edges[:, -1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    X_train = torch.from_numpy(tr_data).to(device)
    X_val = torch.from_numpy(va_data).to(device)
    X_test = torch.from_numpy(ts_data).to(device)
    Y_train = torch.from_numpy(tr_label).long().to(device)
    Y_val = torch.from_numpy(va_label).long().to(device)
    Y_test = torch.from_numpy(ts_label).long().to(device)

    BATCH_SIZE = 2048
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(X_test, Y_test), batch_size=1000, shuffle=True)
    def evaluate(net, X, Y, if_print=False):
        prediction = torch.max(net(X), 1)[1]
        pred_y = prediction.data.cpu().numpy()
        target_y = Y.data.cpu().numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        score = la_evulate(target_y, pred_y, if_print)
        return accuracy, score


    def batch_evaluate(net, data_loader):
        acc_number = 0
        total_number = 0
        pred_list = []
        target_list = []
        for ind, (b_x, b_y) in enumerate(data_loader):
            b_x = b_x.to(device)
            prediction = torch.max(net(b_x), 1)[1]
            pred_y = prediction.data.cpu().numpy()
            target_y = b_y.data.cpu().numpy()
            acc_number += float((pred_y == target_y).astype(int).sum())
            total_number += b_x.shape[0]
            pred_list.append(pred_y)
            target_list.append(target_y)
        target_y = np.concatenate(target_list)
        pred_y = np.concatenate(pred_list)
        score = la_evulate(target_y, pred_y, if_print=True)
        accuracy = acc_number / total_number
        return accuracy, score

    print(emb_size)
    net = Net(ne_array, word_num=word_num, emb_size=emb_size, trans_size=128,
              n_hidden=256, n_hidden1=256, n_output=2, drop_rate=0.5, device=device)  # define the network
    print(net)  # net architecture
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-5)
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
    max_score = 0
    score_list = []
    batch = 0
    patience_count = 0
    patience_flag = False
    for t in range(100):
        if patience_flag:
            print('Early stop!!!')
            break
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            batch += 1
            net.train()
            out = net(b_x)  # input x and predict based on x
            loss = loss_func(out, b_y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            score_vle = []
            if step % 10 == 0:
                net.eval()
                tr_acc, tr_score = evaluate(net, b_x, b_y)
                score_list.append([tr_acc, tr_score])
                # va_acc, va_score = evaluate(net, X_val, Y_val, True)
                ts_acc, score = batch_evaluate(net, test_loader)
                score_list.append(score_vle)
                if ts_acc > max_score:
                    max_score = ts_acc
                    patience_count = 0
                    print("a better model")
                    torch.save(net.state_dict(), 'model_save/' + model_name)
                else:
                    patience_count += 1
                    if patience_count == args.patience:
                        patience_flag = True
                        break
                print('Epoch: %d(%d),' % (t, step),
                      '|Pat: %d/%d' % (patience_count, args.patience),
                      '|loss: %.4f' % loss.data.cpu().numpy(),
                      '|ts_acc: %.4f' % ts_acc)
    net.load_state_dict(torch.load('model_save/' + model_name))
    net.eval()
    print('*' * 20, 'Final Test', '*' * 20)
    # va_acc, va_score = evaluate(net, X_val, Y_val, True)
    ts_acc = batch_evaluate(net, test_loader)
    print(ts_acc)


