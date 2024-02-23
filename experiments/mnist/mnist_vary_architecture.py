"""
EXPERIMENT NAME: Investigating the effect of number of hidden neurons in linear network on MNIST on Hessian eigenspectrum
EXPERIMENT DESCRIPTION: We train multiple neural networks consisting of 1 hidden layer, and varying numbers of hidden nodes,
and then measure the eigenspectrum and RLCT at convergence.
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

    ### PRODUCE LIST OF NETWORKS WITH VARYING SIZES ###
    hidden_nodes = [2, 4, 8, 16, 32]
    neural_nets = []
    for hidden_node in hidden_nodes:
        neural_net = NeuralNet(hidden_nodes=hidden_node, hidden_layers=2).to(device)
        neural_nets.append(neural_net)

    ### HYPERPARAMETERS, LOSS FUNCTION ###
    hyperparams = {
        "lr": 1e-5,
        "batch_size" : 128,
        "num_workers" : 16,
        "num_epochs" : 10,  # MUST BE AT LEAST 5 AS RLCT ESTIMATE TAKES AVERAGE OF LAST 5 EPOCHS
        "momentum" : 0.8,
        "num_draws" : 2000,
        "num_chains" : 1,
        "noise_level" : 1.0,
        "elasticity" : 1000.0,
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
    model_losses = {}
    for neural_net in neural_nets:
        optimizer = t.optim.Adam(neural_net.parameters(), lr=hyperparams["lr"])
        title = f"{optimizer.__class__.__name__} {neural_net.hidden_layers} HL {neural_net.hidden_nodes} HN"
        train_losses = []
        test_losses = []
        print(f"\n======================== {neural_net.hidden_layers} hidden layers, {neural_net.hidden_nodes} hidden nodes in each layer ==========================")
        for epoch in epochs:
            train_loss = train_one_epoch(neural_net, train_loader, optimizer, criterion, device)
            test_loss = evaluate(neural_net, test_loader, criterion, device)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch}/{hyperparams['num_epochs']}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")
        models[title] = neural_net
        model_losses[title] = [train_losses[-1], test_losses[-1]]

    ### PRDOUCE HESSIAN EIGENSPECTRUMS FOR EACH NETWORK ###
    hessians = produce_hessians(models=models,
                                data_loader=test_loader, 
                                num_batches=10,
                                device=device,
                                criterion=criterion)

    ### VISUALISE EIGENSPECTRUM PLOTS IN PLOTLY ###
    figs, eigenspectrum_data = produce_hessian_eigenspectra(hessians, plot_type="log")

    ### CALCULATE ESTIMATE OF NUMBER OF LARGE EIGENVALUES (DIMENSIONS) IN SPECTRUM ###
    hessian_dims = find_hessian_dimensionality(eigenspectrum_data)
    hessian_fig = go.Figure()
    hessian_fig.add_trace(go.Scatter(x=hidden_nodes, y=list(hessian_dims.values()), mode='markers'))
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
    for neural_net in neural_nets:
        results = run_callbacks(train_loader,
                                train_set,
                                model=neural_net,
                                hyperparams=hyperparams,
                                callbacks=[llc_estimator],
                                criterion=criterion["general"],
                                device=device)
        rlct_estimates.append(results["llc/means"][-1]/count_parameters(neural_net))
    rlct_fig = go.Figure()
    rlct_fig.add_trace(go.Scatter(x=hidden_nodes, y=rlct_estimates, mode='markers'))
    rlct_fig.update_layout(
        title=f"Adam RLCT estimation, Elasticity : {hyperparams['elasticity']}, Noise Level : {hyperparams['noise_level']}",
        xaxis_title="Hidden neurons in each layer",
        yaxis_title="RLCT",
    )
    figs.append(rlct_fig)

    ### VISUALISE TRAINING / TESTING LOSS OVER OPTIMISERS ###
    train_test_fig = go.Figure()
    train_test_fig.add_trace(go.Bar(
        x=list(model_losses.keys()),
        y=[loss[0] for loss in model_losses.values()],
        name="Training Losses",
        marker_color="indianred",
    ))
    train_test_fig.add_trace(go.Bar(
        x=list(model_losses.keys()),
        y=[loss[1] for loss in model_losses.values()],
        name="Testing Losses",
        marker_color="lightsalmon",
    ))
    train_test_fig.update_layout(
        title="Training and testing losses of model architectures",
        xaxis_title="Model",
        yaxis_title="Loss",
        barmode="group",
    )
    figs.append(train_test_fig)

    ### PUSH FIGURES TO LOCAL HTML FILE ###
    curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    write_figs_to_html(figs, f".\experiments\mnist\mnist_hidden_nodes_{curr_time}.html", title="Investigating effect of hidden nodes on RLCT / Hessian eigenspectrum")

if __name__ == "__main__":
    freeze_support()    # ONLY REQUIRED FOR WINDOWS, REMOVE IF USING MAC OR LINUX
    main()