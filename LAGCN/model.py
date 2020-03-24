import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, ne_array, word_num, emb_size,
                 trans_size, n_hidden, n_hidden1,
                 n_output, drop_rate, device):
        super(Net, self).__init__()
        self.den_emb = nn.Embedding(word_num, emb_size)
        self.den_emb.weight = nn.Parameter(torch.FloatTensor(ne_array), requires_grad=False)
        self.trans = torch.nn.Linear(emb_size, trans_size).to(device)
        self.hidden = torch.nn.Linear(3 * trans_size, n_hidden).to(device)  # hidden layer
        self.bn = torch.nn.BatchNorm1d(n_hidden, momentum=0.5).to(device)
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden1).to(device)
        self.bn1 = torch.nn.BatchNorm1d(n_hidden1, momentum=0.5).to(device)
        self.dropout = torch.nn.Dropout(drop_rate).to(device)
        self.out = torch.nn.Linear(n_hidden1, n_output).to(device)  # output layer
        self.device = device
        # nn.init.xavier_uniform_(self.trans.weight)
        # nn.init.xavier_uniform_(self.hidden.weight)
        # nn.init.xavier_uniform_(self.hidden1.weight)
        # nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = self.den_emb(x.cpu())
        x = x.to(self.device)
        x = self.trans(x)
        x1 = x[:, 0] + x[:, 1]
        x2 = torch.abs(x[:, 0] - x[:, 1])
        x3 = x[:, 0] * x[:, 1]
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.bn(self.hidden(x)))  # activation function for hidden layer
        x = F.relu(self.bn1(self.hidden1(x)))
        # x = F.relu(self.hidden(x))  # activation function for hidden layer
        # x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
