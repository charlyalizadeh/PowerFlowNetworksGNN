import torch
from torch import tensor
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from metrics import order_merge_sort, order_pearson_corr
import pandas as pd


def train_iteration(model, loader, criterion, optimizer):
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


def test(model, loader, criterion):
    with torch.no_grad():
        model.eval()
        y_true = tensor([]).cuda()
        y_pred = tensor([]).cuda()
        for data in loader:
            data.to('cuda')
            y_pred = torch.cat((y_pred, model(data.x, data.edge_index, data.batch)), 0)
            y_true = torch.cat((y_true, data.y), 0)
        return criterion(y_pred, y_true[:, None]), y_true, y_pred


def evaluate(model, loader, criterion, ax):
    loss, true, pred = test(model, loader, criterion)
    print(f"Loss : {loss}")
    ax.scatter(true.cpu(), true.cpu(), s=3, label='True')
    ax.scatter(true.cpu(), pred.cpu(), s=3, label='Predicted')
    ax.legend()
    return true, pred


def save_experiment(model, test_set, optimizer, criterion,
                    train_loader, val_loader,
                    train_losses, val_losses):
    # Get the results dirname
    results_dir = [str(f) for f in Path("results").iterdir() if f.is_dir()]
    results_dir = Path("results").joinpath(str(len(results_dir)))
    results_dir.mkdir()

    # Generate and save the figs
    fig_train, ax_train = plt.subplots()
    fig_test, ax_test = plt.subplots()

    train_true, train_pred = evaluate(model, train_loader, criterion, ax_train)
    val_true, val_pred = evaluate(model, val_loader, criterion, ax_test)
    train_true = train_true.cpu().detach().numpy()
    train_pred = [i[0] for i in train_pred.cpu().detach().numpy()]
    val_true = val_true.cpu().detach().numpy()
    val_pred = [i[0] for i in val_pred.cpu().detach().numpy()]

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(list(range(len(train_losses))), train_losses)
    ax_loss.plot(list(range(len(val_losses))), val_losses)

    fig_train.savefig(results_dir.joinpath("train.png"))
    fig_test.savefig(results_dir.joinpath("test.png"))
    fig_loss.savefig(results_dir.joinpath("loss.png"))

    # Compute and save the metrics
    train_metrics = pd.DataFrame({
            "r2": r2_score(train_true, train_pred),
            "order_merge_sort": order_merge_sort(train_true, train_pred),
            "order_peasron_corr": order_pearson_corr(train_true, train_pred)
    }, index=[0])
    val_metrics = pd.DataFrame({
            "r2": r2_score(val_true, val_pred),
            "order_merge_sort": order_merge_sort(val_true, val_pred),
            "order_peasron_corr": order_pearson_corr(val_true, val_pred)
    }, index=[0])
    train_metrics.to_csv(results_dir.joinpath("train_metrics.csv"), index=False)
    val_metrics.to_csv(results_dir.joinpath("val_metrics.csv"), index=False)


    # Save the model
    torch.save(model.state_dict(), results_dir.joinpath("model.dict"))
    shutil.copy("model.py", results_dir.joinpath("model.py"))


    # Save the configuration
    optimizer_name = type(optimizer).__name__
    criterion_name = type(criterion).__name__
    config_file = results_dir.joinpath("config.toml")
    with open(config_file, "w") as io:
        io.write(f"optimizer = \"{optimizer_name}\"\n")
        io.write(f"criterion = \"{criterion_name}\"\n")
        io.write(f"test_set = {test_set}")


def train(model, epochs, criterion, optimizer, train_loader, val_loader, train_losses, val_losses):
    print(len(train_loader))
    print(len(val_loader))
    for epoch in range(1, epochs + 1):
        train_iteration(model, train_loader, criterion, optimizer)
        with torch.no_grad():
            train_loss, train_true, train_pred = test(model, train_loader, criterion)
            val_loss, val_true, val_pred = test(model, val_loader, criterion)
            train_losses.append(train_loss.cpu())
            val_losses.append(val_loss.cpu())
            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
