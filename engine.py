"""
Train and eval functions used in train.py
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from approxngd import KFAC

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """"
    Train one epoch of a model.
    `model`: the nn.Module to be trained,
    `train_loader`: the PyTorch DataLoader for the training data,
    `optimizer` : the optimizer class used,
    `criterion` : loss function.
    `device` : whether cuda gpu or cpu
    """
    
    model.train()
    train_loss = 0
    for image, label in tqdm(train_loader):
        image, label = image.to(device), label.to(device)

        # TODO: checks if optimizer is of type KFAC
        if isinstance(optimizer, KFAC):
            model.zero_grad()
            # Estimate with model distribution
            with optimizer.track_forward():
                output = model(image)
                loss = criterion["kfac"](output, label)
            with optimizer.track_backward():
                loss.backward()
            optimizer.update_cov()
            # Compute loss to backprop
            model.zero_grad()
            output = model(image)
            loss = criterion["kfac"](output, label)
            loss.backward()
            optimizer.step(loss=loss)
        else:
            optimizer.zero_grad()
            output = model(image)
            loss = criterion["general"](output, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
    return train_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model with testing data.
    `model` : model to test,
    `test_loader` : PyTorch DataLoader for test data,
    `criterion` : loss function.
    `device` : whether cuda gpu or cpu
    """

    model.eval()
    test_loss = 0
    with t.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion["general"](output, label)
            test_loss += loss.item()
    return test_loss / len(test_loader)


    
