"""
EXPERIMENT NAME: Investigating the eigenspectra, Hessian dimensionality, and RLCT values converged to by different optimisers.
EXPERIMENT DESCRIPTION: We train deep neural networks on the MNIST dataset, and compare their eigenspectra and RLCT for different optimisers.
"""

### IMPORT LIBRARIES ###
from multiprocessing import freeze_support

import os
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

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
from devinterp.slt import sample
from devinterp.slt.llc import OnlineLLCEstimator
from devinterp.slt.wbic import OnlineWBICEstimator

from approxngd import KFAC
from PyHessian.pyhessian import *
from PyHessian.density_plot import *
from general_utils import *
from hessian_utils import *
from models.NN import *

import plotly.express as px
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def main():
    ### CHECK DEVICE ###
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"DEVICE : {device}")
    warnings.filterwarnings("ignore")
    figs = []   # place to store all figures for viewing

    ### HYPERPARAMETERS, MODEL, AND OPTIMISERS ###
    model = NeuralNet(hidden_layers=2, hidden_nodes=256).to(device)
    hyperparams = {
        "lr": 1e-5,
        "batch_size" : 128,
        "num_workers" : 16,
        "num_epochs" : 10,
        "momentum" : 0.8,
        "num_draws" : 4000,
        "num_chains" : 5,
        "noise_level" : 2.0,
        "elasticity" : 10000.0,
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
    optimizers = [sgd, adam, rmsprop, ngd]

    ### DATA LOADING ###
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = t.utils.data.DataLoader(train_set, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=hyperparams["num_workers"], persistent_workers=True)
    test_loader = t.utils.data.DataLoader(test_set, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=hyperparams["num_workers"], persistent_workers=True)

    ### TRAINING LOOP ###
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
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_loss = evaluate(model, test_loader, criterion, device)
            optim_train_losses.append(train_loss)
            optim_test_losses.append(test_loss)
            optim_models.append(model)
            print(f"Epoch {epoch+1}/{hyperparams['num_epochs']}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")
        train_losses[name] = optim_train_losses
        test_losses[name] = optim_test_losses
        models[name] = optim_models
    
    ### GENERATE TRAINING / TESTING PLOTS ###
    train_fig = go.Figure()
    for optim, train_loss in train_losses.items():
        train_fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name=optim))
    train_fig.update_layout(title="Training loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    legend_title="Optimizers"
                    )
    test_fig = go.Figure()
    for optim, test_loss in test_losses.items():
        test_fig.add_trace(go.Scatter(x=epochs, y=test_loss, mode='lines+markers', name=optim))

    test_fig.update_layout(title="Test loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    legend_title="Optimizers",
                    )
    figs.append(train_fig)
    figs.append(test_fig)

    ### COMPUTE MODEL EIGENSPECTRA ###
    hessians = produce_hessians(models={key : value[-1] for key, value in models.items()},
                                data_loader=train_loader,
                                num_batches=10,
                                criterion=criterion,
                                device=device)
    
    ### COMPUTE FIGURES AND EIGENSPECTRUM DATA ###
    hessian_figs, eigenspectrum_data = produce_hessian_eigenspectra(hessians, plot_type="log")
    for hessian_fig in hessian_figs:
        figs.append(hessian_fig)
    
    ### CALCULATE ESTIMATE OF NUMBER OF LARGE EIGENVALUES (DIMENSIONS) IN SPECTRUM ###
    hessian_dims = find_hessian_dimensionality(eigenspectrum_data)
    hessian_fig = go.Figure()
    hessian_fig.add_trace(go.Bar(x=list(models.keys()), y=list(hessian_dims.values())))
    hessian_fig.update_layout(
        title="Hessian dimensionality over models",
        xaxis_title="Hidden neurons",
        yaxis_title="Dimensionality",
    )
    figs.append(hessian_fig)

    ### LLC ESTIMATIONS FOR EACH ARCHITECTURE AT CONVERGENCE ###
    llc_estimator = OnlineLLCEstimator(hyperparams["num_chains"],                                       
                                       hyperparams["num_draws"], 
                                       len(train_set), 
                                       device=device)
    rlct_estimates = []
    for optimiser, neural_nets in models.items():
        results = run_callbacks(train_loader,
                                train_set,
                                model=neural_nets[-1],
                                hyperparams=hyperparams,
                                callbacks=[llc_estimator],
                                criterion=criterion["general"],
                                device=device)
        rlct_estimates.append(results["llc/means"][-1]/count_parameters(neural_nets[-1]))
    rlct_fig = go.Figure()
    rlct_fig.add_trace(go.Bar(x=list(models.keys()), y=rlct_estimates))
    rlct_fig.update_layout(
        title=f"RLCT values for optimisers",
        xaxis_title="Optimiser",
        yaxis_title="RLCT",
    )
    figs.append(rlct_fig)

    ### PUSH FIGURES TO LOCAL HTML FILE ###
    curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    write_figs_to_html(figs, f".\experiments\mnist\mnist_optimisers_{curr_time}.html", title="Investigating effect of optimiser on RLCT / Hessian eigenspectrum")

if __name__ == "__main__":
    freeze_support()
    main()
