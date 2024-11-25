import mlflow
import os
import time
from tqdm import tqdm
from scipy.stats import pearsonr
from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def training_epoch(model, dataloader, optimizer, loss_fn=nn.MSELoss()):
    model.train()
    train_loss = []
    i = 0
    for x, y in tqdm(dataloader):
        mask = torch.sum(x**2, dim=2, keepdim=True) == 0
        x = x.float().to(model.device)
        y = y.float().to(model.device)
        pred, aux = model(x, mask.to(model.device))
        loss = loss_fn(pred[:, 0], y[:, 0])
        train_loss += [loss.detach().cpu().numpy()]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = np.mean(train_loss)
    return train_loss


def predict(model, dataloader):
    model.eval()
    preds = []
    labels = []
    for data in tqdm(dataloader):
        x, y = data
        mask = torch.sum(x**2, dim=2, keepdim=True) == 0
        pred, _ = model(x.float().to(model.device), mask.to(model.device))
        labels += [y[:, 0].numpy()]
        preds += [pred[:, 0].detach().cpu().numpy()]
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return preds, labels


def fit(model,
        train_set,
        params={},
        optimizer=None,
        test_set=None,
        path=None,
        verbose=True,
        mlflow_log_epochs=False,
       ):

    if path is not None and not os.path.exists(path):
        os.mkdir(path)

    default_params = {
        'max_epochs': 20,
        'batch_size': 10,
        'optimizer': 'adam',
        'lr': 1e-3,
        'momentum': 0,
        'num_workers': 0
    }
    default_params.update(params)
    batch_size = default_params['batch_size']
    num_epochs = default_params['max_epochs']
    optimizer = default_params['optimizer']
    lr = default_params['lr']
    momentum = default_params['momentum']
    num_workers = default_params['num_workers']

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(
            list(model.parameters()), lr=lr, weight_decay=0.0
        )
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            list(model.parameters()), lr=lr, momentum=momentum
        )

    start_time = time.time()
    loss_fn = nn.MSELoss()

    try:
        for e in range(num_epochs):

            train_loss = training_epoch(model, train_loader, optimizer, loss_fn)

            if mlflow_log_epochs:
                mlflow.log_metrics(
                    {
                        'train_loss': train_loss,
                        'valid_loss': valid_loss,
                        'pearsonr': R,
                        'p': p
                    },
                    step=e + 1,
                )

            if verbose:
                logger.info('Epoch {}/{} - {:.2f}s'.format(
                    e + 1,
                    num_epochs,
                    time.time() - start_time))
                start_time = time.time()
                logger.info('loss: {:.4f}'.format(train_loss))

    except KeyboardInterrupt:
        pass

    preds, labels = predict(model, test_loader)

    return preds, labels
