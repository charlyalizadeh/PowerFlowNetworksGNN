import torch
from torch import tensor
from PFNDataset import PFNDataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GINConv, DenseGCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch_geometric.transforms as T
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

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


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(47, 16)
        self.conv2 = GCNConv(16, 32)
        self.regressor = Regressor(32, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, 0.5, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.regressor(x)
        return x


model = GCN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.L1Loss()


def train(loader):
    model.train()
    for i, data in enumerate(loader):
        print(f"    {i + 1} / {len(loader)}", end="\r")
        data.to('cuda')
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y[:, None])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print()


def test(loader):
    with torch.no_grad():
        model.eval()
        y_true = tensor([]).cuda()
        y_pred = tensor([]).cuda()
        for data in loader:
            data.to('cuda')
            y_pred = torch.cat((y_pred, model(data.x, data.edge_index, data.batch)), 0)
            y_true = torch.cat((y_true, data.y), 0)
        return criterion(y_pred, y_true[:, None]), y_true, y_pred


def evaluate(loader):
    loss, true, pred = test(loader)
    print(f"Loss : {loss}")
    plt.scatter(true.cpu(), true.cpu(), s=3, label='True')
    plt.scatter(true.cpu(), pred.cpu(), s=3, label='Predicted')
    plt.legend()
    plt.show()
    plt.cla()


train_dataset = PFNDataset('data')
val_dataset = train_dataset.exclude(["case118"])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)
train_losses = []
val_losses = []

for epoch in range(1, 500):
    train(train_loader)
    with torch.no_grad():
        train_loss, train_true, train_pred = test(train_loader)
        #val_loss, val_true, val_pred = test(val_loader)
        train_losses.append(train_loss.cpu())
        #val_losses.append(val_loss.cpu())
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}")


print("Train")
evaluate(train_loader)
val_dataset("Val")
evaluate(val_loader)
plt.plot(list(range(len(train_losses))), train_losses)
#plt.plot(list(range(len(val_losses))), val_losses)
plt.show()
