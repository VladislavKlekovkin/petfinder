import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


# REPRODUCIBILITY
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid_np(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig


def root_mean_square_error(predictions, targets):
    return mean_squared_error(sigmoid_np(predictions) * 100., targets * 100.) ** .5


def get_scheduler(Training, optimizer):
    if Training.scheduler is None:
        return None

    if Training.scheduler is 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=Training.lr,
                                                   steps_per_epoch=1, epochs=Training.epochs)

    if Training.scheduler is 'LambdaLR':
        lr = Training.lr
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: (lr - 0.5*lr*e)/lr)
        # lambda e: .1 ** e

def get_criterion(Training):

    if Training.criterion == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss()


def train_epoch(model, loader, optimizer, criterion, use_meta, device, DEBUG):
    model.train()
    train_loss = []
    PREDICTIONS = []
    TARGETS = []
    bar = tqdm(loader)
    for (data, target) in bar:
        optimizer.zero_grad()
        if use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            predictions = model(data, meta)
        else:
            data, target = data.to(device), target.to(device)
            predictions = model(data)

        PREDICTIONS.append(predictions.detach().cpu())
        TARGETS.append(target.detach().cpu())

        loss = criterion(predictions, target)

        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
        if DEBUG:
            break

    PREDICTIONS = torch.cat(PREDICTIONS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    rmse = root_mean_square_error(PREDICTIONS, TARGETS)

    return train_loss, rmse


def val_epoch(model, loader, criterion, use_meta, device, DEBUG, get_output=False):
    model.eval()
    val_loss = []
    PREDICTIONS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):

            if use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                predictions = model(data, meta)
            else:
                data, target = data.to(device), target.to(device)
                predictions = model(data)

            PREDICTIONS.append(predictions.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(predictions, target)
            val_loss.append(loss.detach().cpu().numpy())
            if DEBUG:
                break

    val_loss = np.mean(val_loss)
    PREDICTIONS = torch.cat(PREDICTIONS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return PREDICTIONS
    else:
        rmse = root_mean_square_error(PREDICTIONS, TARGETS)
        return val_loss, rmse
