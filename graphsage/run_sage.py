from preload import load_data
from model import *
# from edge import *
from utils import *
from sampler import *
from sklearn import metrics
import os

root_path = os.getcwd() + '/'

dataset = 'cora'
batch_size = 256
node_iters = 20
emb_size = 128
node_early_stop = 20
node_least_iter = 100
num_neigh0=5
num_neigh1=5
is_cuda = False
if dataset=='pubmed':
    batch_size=128
    num_neigh0 = 25
    num_neigh1 = 10
    node_least_iter = 80

adj, feature_data, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
adj += sp.diags([1] * adj.shape[0])

# adj = sp.csr_matrix(np.ones_like(adj.todense()))
adj_filt = adj.copy()

# feature_data = nontuple_preprocess_features(feature_data)

tr_idx = np.where(train_mask)[0]
va_idx = np.where(val_mask)[0]
ts_idx = np.where(test_mask)[0]
lb_tr = y_train[train_mask].argmax(axis=1)
lb_va = y_val[val_mask].argmax(axis=1)
lb_ts = y_test[test_mask].argmax(axis=1)
if dataset=='citeseer':
    lb_ts_dict = {ts_idx[i]:lb_ts[i] for i in range(len(ts_idx))}
    new_lb_ts = [lb_ts_dict.get(i,0) for i in range(np.min(ts_idx), np.max(ts_idx)+1)]
    new_lb_ts = np.array(new_lb_ts).reshape([-1])
    lb_oh = one_hot(
        np.concatenate([lb_tr, lb_va, new_lb_ts], axis=0))
else:
    lb_oh = one_hot(
        np.concatenate([lb_tr, lb_va, lb_ts], axis=0))
num_feature = feature_data.shape[1]
num_nodes = adj.shape[0]
num_classes = y_train.shape[1]

s = Sampler(adj)
attack_num=0
print('attack_num%d'%attack_num)
idx_list = np.concatenate([va_idx,ts_idx], axis=0)
adj = att_del_process(s, adj, attack_num, idx_list, lb_oh)
s = Sampler(adj)

sampler = s.normal_sample
features = nn.Embedding(num_nodes, num_feature)
features.weight = nn.Parameter(torch.FloatTensor(feature_data.todense()), requires_grad=False)

agg1 = MeanAggregator(features, sampler, is_softgate=True, cuda=is_cuda)
enc1 = Encoder(features, num_feature, emb_size, agg1, gcn=False, cuda=is_cuda)
agg2 = MeanAggregator(
    lambda nodes: enc1(nodes).t(), sampler, is_softgate=True, cuda=is_cuda)
enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, emb_size, agg2,
    base_model=enc1, gcn=False, cuda=is_cuda)
enc1.num_sample = num_neigh1
enc2.num_sample = num_neigh0
graphsage = SupervisedGraphSage(num_classes, enc2, drop_out=0.5, cuda=is_cuda)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=1.0)

score_list = []
times = []
early_stop_count=0
va_acc_max=0
for batch in range(1000):
    batch_nodes = random.sample(list(tr_idx), batch_size)
    start_time = time.time()
    optimizer.zero_grad()
    label = Variable(torch.LongTensor([lb_tr[i] for i in batch_nodes]))
    if is_cuda:
        label = label.cuda()
    tr_output, loss = graphsage.loss(
        batch_nodes,
        label,
        is_train=True
    )
    loss.backward()
    optimizer.step()
    end_time = time.time()
    times.append(end_time - start_time)
    #         tr_output = graphsage.forward(tr_idx)
    va_output = graphsage.forward(va_idx)
    ts_output = graphsage.forward(ts_idx)
    tr_acc = metrics.accuracy_score(
        [lb_tr[i] for i in batch_nodes],
        tr_output.cpu().data.numpy().argmax(axis=1))
    va_acc = metrics.accuracy_score(
        lb_va, va_output.cpu().data.numpy().argmax(axis=1))
    ts_acc = metrics.accuracy_score(
        lb_ts, ts_output.cpu().data.numpy().argmax(axis=1))
    print('Node Train %d(%d) batchs Tr_acc:%.4f, Va_acc:%.4f, Ts_acc: %.4f' % (
        batch + 1, early_stop_count, tr_acc, va_acc, ts_acc))
    score_list.append((batch, tr_acc, va_acc, ts_acc))
    if va_acc>va_acc_max:
        va_acc_max = va_acc
        early_stop_count=0
    else:
        early_stop_count+=1
        if early_stop_count > node_early_stop and batch > node_least_iter:
            break
score_array = np.array(score_list)
max_ts_acc_iter = score_array[:,3].argmax(axis=0)
max_ts_acc = score_array[max_ts_acc_iter][-1]
max_va_acc_iter = score_array[:,2].argmax(axis=0)
max_va_acc = score_array[max_va_acc_iter][-1]
print('max test acc criterion %.4f' %max_ts_acc )
print(score_array[max_ts_acc_iter])
print('max val acc criterion %.4f' %max_va_acc )
print(score_array[max_va_acc_iter])

print(score_list)
