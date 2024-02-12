"""
Train and eval functions used in train.py
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from approxngd import KFAC

import matplotlib as mpl
#mpl.use('tkagg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PyHessian.density_plot import density_generate

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

def get_esd_plot_plotly(eigenvalues, weights, title=None, fig=None, name=None):
    """
    Produce eigenspectrum plot in Plotly, with support for overlaid plots.
    """

    density, grids = density_generate(eigenvalues, weights)

    if not fig:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grids, y=density + 1.0e-7, mode='lines'))
        fig.update_layout(
            title=title,
            xaxis_title='Eigenvalue',
            yaxis_title='Density (Log Scale)',
            yaxis=dict(type='log'),
        )
        fig.show()
        return fig
    else:
        fig.add_trace(go.Scatter(x=grids, y=density + 1.0e-7, mode='lines', name=name))

def count_parameters(model):
    """
    Given a model, return the total number of parameters.
    """

    return sum(layer.numel() for layer in model.parameters())






    
