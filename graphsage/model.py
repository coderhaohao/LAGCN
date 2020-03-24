import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F
import numpy as np

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, sampler, is_softgate=True, cuda=True, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.sampler = sampler
        self.is_softgate = True
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        nodes = [int(_) for _ in nodes]
        samp_neighs, samp_weights = self.sampler(nodes, num_sample)
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        unique_map = {i: n for i, n in enumerate(unique_nodes_list)}
        weight_matrix = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        if self.is_softgate:
            gate_weight = []
            for i in range(len(column_indices)):
                gate_weight.append(samp_weights[row_indices[i]][unique_map[column_indices[i]]])
            weight_matrix[row_indices, column_indices] = torch.FloatTensor(gate_weight)
        else:
            weight_matrix[row_indices, column_indices] = 1
        if self.cuda:
            weight_matrix = weight_matrix.cuda()
        num_neigh = weight_matrix.sum(1, keepdim=True)
        weight_matrix = weight_matrix.div(num_neigh)
        if self.cuda:
            idx = torch.LongTensor(unique_nodes_list)
            embed_matrix = self.features(idx).cuda()
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = weight_matrix.mm(embed_matrix)
        return to_feats


class AttenMech(nn.Module):
    def __init__(self, emb_size, mid_size=None, cuda=True):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()
        if mid_size is None:
            mid_size = int(emb_size/2)
        self.cuda = cuda
        self.lin = [
            nn.Linear(emb_size, mid_size),
            nn.Linear(2*mid_size, 1)
        ]
        for layer in self.lin:
            init.xavier_uniform_(layer.weight)

    def parameters(self):
        para_list = []
        for i in range(len(self.lin)):
            for p in self.lin[i].parameters():
                if p.requires_grad:
                    para_list.append(p)
        return para_list


    def forward(self, emb_a, emb_b, ):
        emb_a = self.lin[0](emb_a)
        emb_b = self.lin[0](emb_b)
        emb_out = F.leaky_relu_(self.lin[1](F.cat([emb_a, emb_b])))
        return emb_out




class AttentionAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, sampler, am, is_softgate=True, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.sampler = sampler
        self.is_softgate = is_softgate
        self.cuda = cuda
        self.gcn = gcn
        self.am = am

    def forward(self, nodes, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
#         nodes = list(nodes)
        nodes = [int(_) for _ in nodes]
        samp_neighs, samp_weights = self.sampler(nodes, num_sample)
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        unique_map = {i: n for i, n in enumerate(unique_nodes_list)}

        weight_matrix = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))

        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # if self.is_softgate:
        #     gate_weight = []
        #     for i in range(len(column_indices)):
        #         gate_weight.append(samp_weights[row_indices[i]][unique_map[column_indices[i]]])
        #     weight_matrix[row_indices, column_indices] = torch.FloatTensor(gate_weight)
        # else:
        #     weight_matrix[row_indices, column_indices] = 1
        node_a_list = [nodes[i] for i in row_indices]
        node_b_list = [unique_map[column_indices[i]] for i in column_indices]
        if self.cuda:
            node_a_embs = self.features(torch.LongTensor(node_a_list).cuda())
            node_b_embs = self.features(torch.LongTensor(node_b_list).cuda())
        else:
            node_a_embs = self.features(torch.LongTensor(node_a_list))
            node_b_embs = self.features(torch.LongTensor(node_a_list))
        gate_weight = self.am(node_a_embs, node_b_embs)
        weight_matrix[row_indices, column_indices] = gate_weight
        if self.cuda:
            weight_matrix = weight_matrix.cuda()
        num_neigh = weight_matrix.sum(1, keepdim=True)
        weight_matrix = weight_matrix.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = weight_matrix.mm(embed_matrix)
        return to_feats


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim,
            embed_dim, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False,
            feature_transform=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        if cuda:
            self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim).cuda())
        else:
            self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, self.num_sample)
        if not self.gcn:
            if self.cuda:
                idx = torch.LongTensor(nodes)
                self_feats = self.features(idx).cuda()
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined


class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, enc, drop_out=0.5, cuda=False):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        if cuda:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim).cuda())
        else:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, nodes, is_train=False):
        embeds = self.enc(nodes)
        if is_train:
            embeds = self.dropout(embeds)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels, is_train=False):
        scores = self.forward(nodes, is_train)
        return scores, self.xent(scores, labels.squeeze())

    def embed(self, nodes):
        embeds = self.enc(nodes)
        return embeds.t()

    def embed_predict(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds).t()
        soft = F.softmax(scores, dim=1)
        return embeds.t(), scores, soft


class EdgePredictor(nn.Module):
    def __init__(self, emb_size, nb_classes=2, drop_rate=0.5):
        super(EdgePredictor, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.lin0 = nn.Linear(3 * emb_size, int(emb_size))
        self.lin1 = nn.Linear(int(emb_size), nb_classes)
        init.xavier_uniform_(self.lin0.weight)
        init.xavier_uniform_(self.lin1.weight)

    def forward(self, features, nodes, is_train=False):
        nodes_a = features(torch.LongTensor(nodes[:,0]))
        nodes_b = features(torch.LongTensor(nodes[:,1]))
        nodes_aplusb = nodes_a + nodes_b
        nodes_ab = nodes_a * nodes_b
        nodes_aminusb = torch.abs_(nodes_a - nodes_b)
        con1 = torch.cat([nodes_aplusb, nodes_ab, nodes_aminusb], 1)
        if torch.cuda.is_available():
            con1=con1.cuda()
        fc1 = torch.sigmoid(self.lin0(con1))
        if is_train == True:
            fc1 = self.dropout(fc1)
        scores = self.lin1(fc1)
        return scores

    def loss(self, features, nodes, labels, is_train):
        scores = self.forward(features, nodes, is_train)
        loss = self.criterion(scores, labels)
        return scores, loss

def forward(model, nodes, batch_size=256):
    out_list = []
    for i in range(0,len(nodes),batch_size):
        j = min(i+batch_size, len(nodes))
        out = model.forward(nodes[i:j]).cpu().data.numpy()
        out_list.append(out)
    return np.vstack(out_list)