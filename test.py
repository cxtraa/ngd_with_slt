# Import required libraries

import os
import sys
import warnings
import numpy as np
import einops

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

from tqdm import tqdm
from datetime import datetime
import json
import wandb

from devinterp.slt import estimate_learning_coeff
from devinterp.optim.sgld import SGLD

from approxngd import KFAC
from PyHessian.pyhessian import hessian
from PyHessian.density_plot import *

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

#device = "cuda" if t.cuda.is_available() else "cpu"
device = "cuda"
warnings.filterwarnings("ignore")

#%%

# Define MNIST model architecture

class NeuralNet(nn.Module):
    """
    Simple template NN architecture.
    Adjust architecture accordingly for experiment.
    """
    
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.model(x)

model = NeuralNet().to(device)
print(model)

#%%
# Define model hyperparameters, loss function, and optimizers

hyperparams = {
    "lr": 1e-5,
    "batch_size" : 128,
    "num_epochs" : 5,  # MUST BE AT LEAST 5 AS RLCT ESTIMATE TAKES AVERAGE OF LAST 5 EPOCHS
    "momentum" : 0.8,
    "num_draws" : 400,
    "num_chains" : 1,
    "noise_level" : 0.5,
    "elasticity" : 50,
}

epochs = np.arange(1, hyperparams["num_epochs"]+1)

criterion = {"general":nn.CrossEntropyLoss(),"kfac": nn.CrossEntropyLoss(reduction='mean')}
sgd = t.optim.SGD(
    model.parameters(),
    lr=hyperparams["lr"],
    )
adam = t.optim.Adam(
    model.parameters(),
    lr=hyperparams["lr"],
    )
rmsprop = t.optim.RMSprop(
    model.parameters(),
    lr=hyperparams["lr"],
    momentum=hyperparams["momentum"],
)
ngd = KFAC(model, 
           hyperparams["lr"], 
           1e-3,
           momentum_type='regular',
           momentum=hyperparams["momentum"],
           adapt_damping=False,
           update_cov_manually=True,
           )
optimizers = [adam, rmsprop, ngd]

#%%

# Load MNIST data

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = t.utils.data.DataLoader(train_set, batch_size=hyperparams["batch_size"], shuffle=True)
test_loader = t.utils.data.DataLoader(test_set, batch_size=hyperparams["batch_size"], shuffle=False)

#%%

# Define training and evaluation functions

def train_one_epoch(model, train_loader, optimizer, criterion):
    """"
    Train one epoch of a model.
    `model`: the nn.Module to be trained,
    `train_loader`: the PyTorch DataLoader for the training data,
    `optimizer` : the optimizer class used,
    `criterion` : loss function.
    """
    
    model.train()
    train_loss = 0
    for image, label in tqdm(train_loader):
        image, label = image.to(device), label.to(device)
        if optimizer == ngd:
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

def evaluate(model, test_loader, criterion):
    """
    Evaluate the model with testing data.
    `model` : model to test,
    `test_loader` : PyTorch DataLoader for test data,
    `criterion` : loss function.
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

#%% main loop
# For each optimiser, train the model and record train and test losses.

models = {}
train_losses = {}
test_losses = {}
for optimizer in optimizers:
    name = f"{optimizer.__class__.__name__}"
    optim_models = []
    optim_train_losses = []
    optim_test_losses = []
    print(f"\n======================== Training with {name} ==========================")
    for epoch in range(hyperparams["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss = evaluate(model, test_loader, criterion)
        optim_train_losses.append(train_loss)
        optim_test_losses.append(test_loss)
        optim_models.append(model)
        print(f"Epoch {epoch+1}/{hyperparams['num_epochs']}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")
    train_losses[name] = optim_train_losses
    test_losses[name] = optim_test_losses
    models[name] = optim_models