"""
EXPERIMENT NAME: Investigating the effect of number of hidden neurons in linear network on MNIST on Hessian eigenspectrum
EXPERIMENT DESCRIPTION: We train multiple LinearNNs consisting of a FIXED number of hidden layers,
and varying numbers of hidden nodes,
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
from utils_general import *
from utils_hessian import *
from utils_rlct import *
from architectures.Linear import *
from data.build_data import build_data

import plotly.express as px
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def get_mnist_vary_HN_args_parser():
    parser = argparse.ArgumentParser(description='Set parameters for training model', add_help=False)

    # Arguments organized by group
    arg_groups = {
        'RLCT Hyperparameters': [
            {'name': '--num_draws', 'default': 1000, 'type': int},
            {'name': '--num_chains', 'default': 2, 'type': int},
            {'name': '--epsilon', 'default': 1e-5, 'type':float,'help':'This is the LLC step size'},
            {'name': '--gamma', 'default': 1, 'type':float,'help':'This is localization factor'}
        ],
        'Hessian Parameters': [
            {'name': '--hessian_batch_size', 'default': 12, 'type': int},
        ],
        'Data Loading Parameters': [
            {'name': '--batch_size', 'default': 128, 'type': int},
            {'name': '--num_workers', 'default': 12, 'type': int},

        ],
        'Selection Criteria':[
            #change your selection criteria here
            {'name': '--criteria', 
             'default': {
                 'model':'LM',
                 'optimiser':'adam',
                 #List of available hidden nodes
                 'LMHN':[2,4,8],
                 #fixed number of hidden layers
                 'LMHL':2
             }, 
             'type': dict},
        ]
    }

    # Loop through the argument groups and add them to the parser
    for group_name, args in arg_groups.items():
        group = parser.add_argument_group(group_name)
        for arg in args:
            group.add_argument(arg['name'], default=arg['default'], type=arg['type'], help=arg.get('help', ''))

    return parser


def main(args):
    ### CHECK DEVICE ###
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"DEVICE : {device}")
    warnings.filterwarnings("ignore")

    ### PRODUCE LIST OF NETWORKS WITH VARYING SIZES ###
    hidden_nodes=args.criteria['LMHN']
    hidden_layers=args.criteria['LMHL']

    models = {}
    titles = []

    for hidden_node in hidden_nodes:
        title = f"{hidden_node} HN {hidden_layers} HL"
        titles.append(title)
        model = LinearMNIST(hidden_nodes=hidden_node, hidden_layers=hidden_layers).to(device)
        models[title] = model

    ### LOAD MODELS FROM LOCAL FILES###
    #this loads in all model histories for one model
    models_histories, models_data = load_models("./weights", criteria=args.criteria)
    #select only the last epoch for each model architecture
    state_dicts = [history[-1] for history in models_histories]
    assert (len(state_dicts)>1), "Check if your criteria is correct!"
    
    num_epochs = models_data[0]["description"]["num_epochs"]
    epochs = np.arange(1, num_epochs+1)


    for i in range(len(state_dicts)):
        HN, HL = models_data[i]["description"]["LMHN"], models_data[i]["description"]["LMHL"]
        optim = models_data[i]["description"]["optimiser"]
        title = f"{HN} HN {HL} HL"
        models[title].load_state_dict(state_dicts[i])

    metric = {"general":nn.CrossEntropyLoss(),"kfac": nn.CrossEntropyLoss(reduction='mean')}
    train_loader, test_loader = build_data(args)
        
    ### PRDOUCE HESSIAN EIGENSPECTRUMS FOR EACH NETWORK ###
    hessians = produce_hessians(models=models,
                                # FIXME: compute hessian on the train_set here?
                                data_loader=train_loader, 
                                num_batches=args.hessian_batch_size,
                                device=device,
                                criterion=metric,
                                history=False)

    ### VISUALISE EIGENSPECTRUM PLOTS IN PLOTLY ###
    figs, eigenspectrum_data = produce_hessian_eigenspectra(hessians, plot_type="log",history=False)

    ### CALCULATE ESTIMATE OF NUMBER OF LARGE EIGENVALUES (DIMENSIONS) IN SPECTRUM ###
    hessian_dims, hessian_dims_norm = produce_hessian_dimensionality(eigenspectrum_data,history=False)
    hessian_fig = go.Figure()
    hessian_fig.add_trace(go.Scatter(x=hidden_nodes, y=list(hessian_dims.values()), mode='markers'))
    hessian_fig.update_layout(
        title="Hessian dimensionality over models",
        xaxis_title="Hidden layers",
        yaxis_title="Dimensionality",
    )
    figs.append(hessian_fig)

    ### ESTIMATE RLCT
    rlct_estimates, rlct_estimates_norm, neg_log_likelyhoods = produce_rlct(models, train_loader,metric, device, args,history=False)

    rlct_fig = go.Figure()
    Y = [rlct_estimates[f"{hn} HN {hidden_layers} HL"] for hn in hidden_nodes]
    #rlct_fig.add_trace(go.Scatter(x=hidden_nodes, y=Y, mode='markers'))
    rlct_fig.add_trace(go.Scatter(x=hidden_nodes, y=Y))
    rlct_fig.update_layout(
        title=f"Adam RLCT estimation, epsilon : {args.epsilon}, gamma : {args.gamma}",
        xaxis_title="Hidden neurons in each layer",
        yaxis_title="RLCT",
    )
    figs.append(rlct_fig)

    rlct_fig_norm = go.Figure()
    Y_norm = [rlct_estimates_norm[f"{hn} HN {hidden_layers} HL"] for hn in hidden_nodes]
    rlct_fig_norm.add_trace(go.Scatter(x=hidden_nodes, y=Y_norm))
    rlct_fig_norm.update_layout(
        title=f"Adam RLCT_norm estimation, epsilon : {args.epsilon}, gamma : {args.gamma}",
        xaxis_title="Hidden neurons in each layer",
        yaxis_title="RLCT_norm",
    )
    figs.append(rlct_fig_norm)

    ### VISUALISE TRAINING / TESTING / GENERALIZATION LOSS OVER MODEL ARCHITECTURES ###

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Bar(
        x=titles,
        y=[model_data["train_losses"][-1] for model_data in models_data],
        name="Training Losses",
        marker_color="indianred",
    ))
    loss_fig.add_trace(go.Bar(
        x=titles,
        y=[model_data["test_losses"][-1] for model_data in models_data],
        name="Testing Losses",
        marker_color="lightsalmon",
    ))

    loss_fig.add_trace(go.Bar(
        x=titles,
        y=[neg_log_likelyhoods[title] - rlct_estimates[title]/args.num_draws for title,model in models.items()],
        name="Generalization Losses",
        marker_color="mediumseagreen",
    ))
    loss_fig.update_layout(
        title="Training, testing and generalisation losses of model architectures",
        xaxis_title="Model",
        yaxis_title="Loss",
        barmode="group",
    )
    figs.append(loss_fig)


    ### PUSH FIGURES TO LOCAL HTML FILE ###
    curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    write_figs_to_html(figs, f"./experiments/mnist/figs/mnist_hidden_nodes_{curr_time}.html", title="Investigating effect of hidden conv layers on CNN Hessian eigenspectrum")

if __name__ == "__main__":
    freeze_support()    # ONLY REQUIRED FOR WINDOWS, REMOVE IF USING MAC OR LINUX
    parser = get_mnist_vary_HN_args_parser()
    args = parser.parse_args()
    main(args)