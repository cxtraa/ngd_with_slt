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
from architectures.NN import *

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def main():
    ### CHECK DEVICE ###
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"DEVICE : {device}")
    warnings.filterwarnings("ignore")

    ### PRODUCE MULTIPLE MODELS FOR TRAINING WITH DIFFERENT OPTIMISERS ###
    HIDDEN_LAYERS = 2
    HIDDEN_NODES = 128
    optimisers = ["sgd", "rmsprop", "adam", "ngd"]
    models = {}
    for optim in optimisers:
        model = LinearMNIST(hidden_layers=HIDDEN_LAYERS, hidden_nodes=HIDDEN_NODES).to(device)
        models[optim] = model

    ### HYPERPARAMETERS, MODEL, AND OPTIMISERS ###
    hyperparams = {
        "lr": 1e-5,
        "batch_size" : 128,
        "num_epochs" : 10,
        "num_workers" : 12,
        "num_draws" : 2000,
        "num_chains" : 1,
        "noise_level" : 2.0,
        "elasticity" : 1000.0,
    }
    criterion = {"general":nn.CrossEntropyLoss(),"kfac": nn.CrossEntropyLoss(reduction='mean')}

    ### DATA LOADING ###
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = t.utils.data.DataLoader(train_set, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=hyperparams["num_workers"], persistent_workers=True)
    test_loader = t.utils.data.DataLoader(test_set, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=hyperparams["num_workers"], persistent_workers=True)

    ### LOAD MODELS FROM LOCAL FILES ###
    criteria = {
        "model" : "LM",
        "optimiser" : ["adam", "rmsprop", "ngd", "sgd"],
        "LMHN" : HIDDEN_NODES,
        "LMHL" : HIDDEN_LAYERS,
    }
    state_dicts, models_data = load_models("./saved_models", criteria=criteria)
    num_epochs = models_data[0]["description"]["num_epochs"]
    epochs = np.arange(1, num_epochs+1)

    for i in range(len(state_dicts)):
        optim = models_data[i]["description"]["optimiser"]   
        models[optim].load_state_dict(state_dicts[i])

    ### COMPUTE MODEL EIGENSPECTRA ###
    hessians = produce_hessians(models=models,
                                data_loader=test_loader,
                                num_batches=10,
                                criterion=criterion,
                                device=device)
    
    ### COMPUTE FIGURES AND EIGENSPECTRUM DATA ###
    figs, eigenspectrum_data = produce_hessian_eigenspectra(hessians, plot_type="log")
    
    ### CALCULATE ESTIMATE OF NUMBER OF LARGE EIGENVALUES (DIMENSIONS) IN SPECTRUM ###
    hessian_dims, hessian_dims_norm = find_hessian_dimensionality(eigenspectrum_data)
    hessian_dims_fig = go.Figure()
    hessian_dims_fig.add_trace(go.Bar(
        x=optimisers,
        y=list(hessian_dims.values()),
        name="Dims (Raw)",
    ))
    hessian_dims_fig.add_trace(go.Bar(
        x=optimisers,
        y=list(hessian_dims_norm.values()),
        name="Dims (Normalised)",
    ))
    hessian_dims_fig.update_layout(
        title="Hessian dimensionality over optimisers",
        xaxis_title="Optimiser",
        yaxis_title="Hessian dimensions",
    )
    figs.append(hessian_dims_fig)

    ### VISUALISE TRAINING / TESTING LOSS OVER OPTIMISERS ###
    train_test_fig = go.Figure()
    train_test_fig.add_trace(go.Bar(
        x=[model_data["description"]["optimiser"] for model_data in models_data],
        y=[model_data["train_losses"][-1] for model_data in models_data],
        name="Training Losses",
        marker_color="indianred",
    ))
    train_test_fig.add_trace(go.Bar(
        x=[model_data["description"]["optimiser"] for model_data in models_data],
        y=[model_data["test_losses"][-1] for model_data in models_data],
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

    train_fig = go.Figure()
    test_fig = go.Figure()
    for model_data in models_data:
        train_fig.add_trace(go.Scatter(
            x=epochs,
            y=model_data["train_losses"],
            name=model_data["description"]["optimiser"],
        ))
        test_fig.add_trace(go.Scatter(
            x=epochs,
            y=model_data["test_losses"],
            name=model_data["description"]["optimiser"],
        ))
    train_fig.update_layout(
        title="Evolution of training loss over optimisers",
        xaxis_title="Epochs",
        yaxis_title="Loss",
    )
    test_fig.update_layout(
        title="Evolution of testing loss over optimisers",
        xaxis_title="Epochs",
        yaxis_title="Loss",
    )
    figs.append(train_fig)
    figs.append(test_fig)

    ### LLC ESTIMATIONS FOR EACH ARCHITECTURE AT CONVERGENCE ###
    llc_estimator = OnlineLLCEstimator(hyperparams["num_chains"],                                       
                                       hyperparams["num_draws"], 
                                       len(train_set), 
                                       device=device)
    rlct_estimates = {}
    rlct_estimates_norm = {}
    for optimiser, model in models.items():
        print(type(model))
        
        results = run_callbacks(train_loader=train_loader,
                                model=model,
                                hyperparams=hyperparams,
                                callbacks=[llc_estimator],
                                criterion=criterion["general"],
                                device=device)
        rlct_estimates[optimiser] = results["llc/means"][-1]
        rlct_estimates_norm[optimiser] = results["llc/means"][-1]/count_parameters(model)
    rlct_fig = go.Figure()
    rlct_fig.add_trace(go.Bar(
        x=list(rlct_estimates.keys()),
        y=list(rlct_estimates.values()),
        name="RLCT (Raw)"
    ))
    rlct_fig.add_trace(go.Bar(
        x=list(rlct_estimates_norm.keys()),
        y=list(rlct_estimates_norm.values()),
        name="RLCT (Normalised)"
    ))
    rlct_fig.update_layout(
        title=f"RLCT values for optimisers",
        xaxis_title="Optimiser",
        yaxis_title="RLCT",
    )
    figs.append(rlct_fig)

    ### PUSH FIGURES TO LOCAL HTML FILE ###
    curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    write_figs_to_html(figs, f"./experiments/mnist/figs/mnist_optimisers_{curr_time}.html", title="Investigating effect of optimiser on RLCT / Hessian eigenspectrum")

if __name__ == "__main__":
    freeze_support()
    main()
