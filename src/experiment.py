import torch
from PFNDataset import PFNDataset, display_dataset
from torch_geometric.loader import DataLoader
from model import GNN
from utils import train, save_experiment


torch.cuda.empty_cache()

# Config
test_set = ["case1888rte", "case6515rte", "case2746wp"]
model = GNN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
epochs = 100

# Dataset
train_dataset = PFNDataset('../data')
val_dataset = train_dataset.exclude(test_set)
display_dataset(train_dataset)
display_dataset(val_dataset)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

# Save losses
train_losses = []
val_losses = []

# Train
train(model, epochs, criterion, optimizer, train_loader, val_loader, train_losses, val_losses)

# Save the experiment
save_experiment(model, test_set, optimizer, criterion,
                train_loader, val_loader,
                train_losses, val_losses)
