import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GINConv, DenseGCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool


class Regressor(torch.nn.Module):
    def __init__(self, nb_in, nb_out):
        super(Regressor, self).__init__()
        self.lin1 = Linear(nb_in, 32)
        self.lin2 = Linear(32, 16)
        self.lin3 = Linear(16, 8)
        self.lin4 = Linear(8, nb_out)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        x = x.relu()
        x = self.lin4(x)
        return x


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(47, 64)
        self.conv2 = GCNConv(64, 32)
        self.regressor = Regressor(32, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, 0.5, training=self.training)
        x = global_add_pool(x, batch)
        x = self.regressor(x)
        return x
