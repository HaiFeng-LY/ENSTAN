import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GATConv(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout, device):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device


        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_layer = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, adj):

        N = x.size()[1]
        outputs = []
        for head in range(self.num_heads):

            Wh = torch.matmul(x, self.W[head])
            Wh_repeated_in_dim1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)
            Wh_repeated_in_dim2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)
            a_input = torch.cat([Wh_repeated_in_dim1,Wh_repeated_in_dim2],dim=-1)
            a_input = a_input.view(x.size(0) * N * N, -1)
            e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(1))
            e = e.view(x.size(0), N, N)
            attention = self.softmax(e)

            attention = attention * adj
            attention = self.dropout_layer(attention)
            h_prime = torch.matmul(attention, Wh)
            outputs.append(h_prime)

        output = torch.cat(outputs, dim=-1)
        return output


class DyGAT(nn.Module):
    def __init__(self, in_channels, num_electrodes, num_heads, out_channels, hidden_dim, A_init, dropout, device, num_classes=4):
        super(DyGAT, self).__init__()
        self.num_heads = num_heads
        self.layer1 = GATConv(in_channels, out_channels, num_heads, dropout, device)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc1 = nn.Linear(num_electrodes * num_heads * out_channels, num_electrodes * num_heads * out_channels)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).to(device))

        self.A = nn.Parameter(torch.tensor(A_init, dtype=torch.float32).cuda())
        self.A.requires_grad = True

    def reset_parameters(self):

        if hasattr(self.fc1, "reset_parameters"):
            self.fc1.reset_parameters()
        if hasattr(self.fc2, "reset_parameters"):
            self.fc2.reset_parameters()
        if hasattr(self.layer1, "reset_parameters"):
            self.layer1.reset_parameters()

    def update_adjacency_matrix(self, total_grad, learning_rate):

        with torch.no_grad():
            rho = learning_rate
            self.A.data = (1 -rho) * self.A.data + rho * total_grad

            self.A.grad.zero_()

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        adj = self.A.unsqueeze(0).repeat(x.size(0), 1, 1)

        result = self.layer1(x, adj)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result = F.dropout(result, p=0.1)
        result = result.reshape(x.shape[0], x.shape[1],x.shape[2])
        return result



class DMGAT(nn.Module):
    def __init__(self, in_channels, num_ele_eeg_fnirs,num_ele_hbo_hbr, num_heads,
                 out_channels, hidden_dim,A_init_eeg_fnirs,A_init_hbo_hbr, dropout, devices, num_classes=4):
        super(DMGAT, self).__init__()
        self.gat_eo = DyGAT(in_channels,num_ele_eeg_fnirs,num_heads, out_channels,hidden_dim,A_init_eeg_fnirs, dropout, devices, num_classes=4)
        self.gat_er = DyGAT(in_channels,num_ele_eeg_fnirs,num_heads, out_channels,hidden_dim,A_init_eeg_fnirs, dropout, devices, num_classes=4)
        self.gat_ro = DyGAT(in_channels,num_ele_hbo_hbr,num_heads, out_channels,hidden_dim,A_init_hbo_hbr, dropout, devices, num_classes=4)
        # print(A_init_hbo_hbr)
    def forward(self, eo,er,ro):
        eo, er , ro = eo.float(), er.float(), ro.float()
        eo = self.gat_eo(eo)
        er = self.gat_er(er)
        ro = self.gat_ro(ro)
        # print('ro',ro.size())
        x_eeghbo = eo.view(-1, 80, 200)
        x_eeg1 = x_eeghbo[:, :62, :]
        x_hbo1 = x_eeghbo[:, 62:, :]

        x_eeghbr = er.view(-1, 80, 200)
        x_eeg2 = x_eeghbr[:, :62, :]
        x_hbr1 = x_eeghbr[:, 62:, :]

        x_hbohbr = ro.view(-1, 36, 200)
        x_hbo2 = x_hbohbr[:, :18, :]
        x_hbr2 = x_hbohbr[:, 18:, :]

        x_eeg = x_eeg1 + x_eeg2
        x_hbo = x_hbo1 + x_hbo2
        x_hbr = x_hbr1 + x_hbr2
        x_eeg = x_eeg.permute(0, 2, 1)
        x_hbo = x_hbo.permute(0, 2, 1)
        x_hbr = x_hbr.permute(0, 2, 1)
        return x_eeg,x_hbo,x_hbr