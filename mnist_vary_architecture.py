"""
EXPERIMENT NAME: Investigating the effect of number of hidden neurons in linear network on MNIST on Hessian eigenspectrum
EXPERIMENT DESCRIPTION: We train multiple neural networks consisting of 1 hidden layer, and varying numbers of hidden nodes,
and then measure the eigenspectrum and RLCT at convergence.
"""

### IMPORT LIBRARIES ###
from multiprocessing import freeze_support # !!!!!!!!!! REMOVE IF USING MAC OR LINUX, THIS IS FOR WINDOWS ONLY !!!!!!!!!!! #

import os
import sys
import warnings
import numpy as np
import pandas as pd
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
from PyHessian.pyhessian import *
from PyHessian.density_plot import *
from eval import *
from engine import *
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

    ### PRODUCE LIST OF NETWORKS WITH VARYING SIZES ###
    neurons = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    neural_nets = []
    for hidden_nodes in neurons:
        neural_net = NeuralNet(hidden_nodes=hidden_nodes, hidden_layers=2).to(device)
        neural_nets.append(neural_net)

    ### HYPERPARAMETERS, LOSS FUNCTION ###
    hyperparams = {
        "lr": 1e-5,
        "batch_size" : 128,
        "num_workers" : 16,
        "num_epochs" : 10,  # MUST BE AT LEAST 5 AS RLCT ESTIMATE TAKES AVERAGE OF LAST 5 EPOCHS
        "momentum" : 0.8,
        "num_draws" : int(1e4),
        "num_chains" : 1,
        "noise_level" : 0.5,
        "elasticity" : 100,
    }
    epochs = np.arange(1, hyperparams["num_epochs"]+1)
    criterion = {"general":nn.CrossEntropyLoss(),"kfac": nn.CrossEntropyLoss(reduction='mean')}

    ### LOAD MNIST DATA ###
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
    for neural_net in neural_nets:
        optimizer = t.optim.Adam(neural_net.parameters(), lr=hyperparams["lr"])
        train_losses = []
        test_losses = []
        print(f"\n======================== {neural_net.hidden_layers} hidden layers, {neural_net.hidden_nodes} hidden nodes in each layer ==========================")
        for epoch in epochs:
            train_loss = train_one_epoch(neural_net, train_loader, optimizer, criterion, device)
            test_loss = evaluate(neural_net, test_loader, criterion, device)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch}/{hyperparams['num_epochs']}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")
        models[f"{neural_net.hidden_layers} hidden layers {neural_net.hidden_nodes} hidden nodes with {optimizer.__class__.__name__}"] = neural_net

    ### PRDOUCE HESSIAN EIGENSPECTRUMS FOR EACH NETWORK ###
    hessians = produce_hessians(models=models,
                                data_loader=test_loader, 
                                num_batches=10,
                                device=device,
                                criterion=criterion)

    ### VISUALISE EIGENSPECTRUM PLOTS IN PLOTLY ###
    figs = produce_hessian_eigenspectra(hessians, plot_type="log")

    ### PERFORM LOCAL LEARNING COEFFICIENT (LLC) ESTIMATION AT CONVERGENCE ###
    print("==================== ESTIMATING LOCAL LEARNING COEFFICIENT FOR MODELS ====================")
    rlct_estimates = estimate_rlcts(models=list(models.values()),
                                    data_loader=train_loader,
                                    criterion=criterion["general"],
                                    data_length=len(train_set),
                                    device=device,
                                    method="SGLD",
                                    method_kwargs={"lr" : 1e-5, "elasticity" : 100.0},
                                    chains=hyperparams["num_chains"],
                                    draws=hyperparams["num_draws"])
    
    ### PLOT RLCT VALUES AGAINST HIDDEN NODE NUMBER ###
    rlct_fig = go.Figure()
    rlct_fig.add_trace(go.Scatter(x=neurons, y=rlct_estimates, mode="markers"))
    rlct_fig.update_layout(
        title="RLCT values over different hidden layer sizes",
        xaxis_title="No. of hidden nodes in each layer",
        yaxis_title="RLCT estimate",
    )   
    figs.append(rlct_fig) 

    ### PUSH FIGURES TO LOCAL HTML FILE ###
    write_figs_to_html(figs, "mnist_hidden_nodes.html", title="Investigating effect of hidden nodes on RLCT / Hessian eigenspectrum")

if __name__ == "__main__":
    freeze_support()    # ONLY REQUIRED FOR WINDOWS, REMOVE IF USING MAC OR LINUX
    main()