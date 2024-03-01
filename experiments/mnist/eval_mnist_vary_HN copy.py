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
from general_utils import *
from hessian_utils import *
from networks import *
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
            {'name': '--noise_level', 'default': 1.0, 'type': float},
            {'name': '--elasticity', 'default': 1000.0, 'type': float}
        ],
        'Hessian Parameters': [
            {'name': '--hessian_batch_size', 'default': 12, 'type': int},
        ],
        'Data Loading Parameters': [
            {'name': '--batch_size', 'default': 128, 'type': int},
            {'name': '--num_workers', 'default': 12, 'type': int},

        ],
        'Selection Criteria':[
            {'name': '--hidden_nodes', 'default': [2,4,8], 'type': list},
            #fixed number of hidden layers
            {'name': '--hidden_layers', 'default': 2, 'type': int},
            {'name': '--model', 'default': "LM", 'type': str},
            {'name': '--optimiser', 'default': "adam", 'type': str},
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
<<<<<<< HEAD:experiments/mnist/eval_mnist_vary_HN.py

    models = {}
    titles = []
    for hidden_node in args.hidden_nodes:
        title = f"{hidden_node} HN {args.hidden_layers} HL"
        titles.append(title)
        model = LinearMNIST(hidden_nodes=hidden_node, hidden_layers=args.hidden_layers).to(device)
=======
    #hidden_nodes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    hidden_nodes = [512, 1024, 2048]
    hidden_layers = 2
    models = {}
    titles = []
    for hidden_conv_layer in hidden_conv_layers:
        title = f"{kernel_size} KS {hidden_conv_layer} HL {optim} optimiser"
        titles.append(title)
        model = CnnMNIST(hidden_conv_layers=hidden_conv_layer).to(device)
>>>>>>> 8dae476ab524a31dd5e5b5b4bbea17a2c7c654a2:experiments/mnist/eval_mnist_vary_architecture.py
        models[title] = model

    ### LOAD MODELS FROM LOCAL FILES ###
    criteria = {
<<<<<<< HEAD:experiments/mnist/eval_mnist_vary_HN.py
        "model" : args.model,
        "optimiser" : args.optimiser,
        "LMHN" : args.hidden_nodes,
        "LMHL" : args.hidden_layers,
=======
        "model" : "CM",
        "optimiser" : "sgd",
        "CMHL" : hidden_conv_layers,
        "KS" : kernel_size,
>>>>>>> 8dae476ab524a31dd5e5b5b4bbea17a2c7c654a2:experiments/mnist/eval_mnist_vary_architecture.py
    }

    state_dicts, models_data = load_models("./weights", criteria=criteria)
    num_epochs = models_data[0]["description"]["num_epochs"]
    epochs = np.arange(1, num_epochs+1)

    for i in range(len(state_dicts)):
        KS, HL = models_data[i]["description"]["KS"], models_data[i]["description"]["CMHL"]
        optim = models_data[i]["description"]["optimiser"]
        title = f"{KS} KS {HL} HL {optim} optimiser"        
        models[title].load_state_dict(state_dicts[i])

    criterion = {"general":nn.CrossEntropyLoss(),"kfac": nn.CrossEntropyLoss(reduction='mean')}
    train_loader, test_loader = build_data(args)
        
    ### PRDOUCE HESSIAN EIGENSPECTRUMS FOR EACH NETWORK ###
    hessians = produce_hessians(models=models,
                                # FIXME: compute hessian on the train_set here?
                                data_loader=train_loader, 
                                num_batches=args.hessian_batch_size,
                                device=device,
                                criterion=criterion)

    ### VISUALISE EIGENSPECTRUM PLOTS IN PLOTLY ###
    figs, eigenspectrum_data = produce_hessian_eigenspectra(hessians, plot_type="log")

    ### CALCULATE ESTIMATE OF NUMBER OF LARGE EIGENVALUES (DIMENSIONS) IN SPECTRUM ###
    hessian_dims, hessian_dims_norm = find_hessian_dimensionality(eigenspectrum_data)
    hessian_fig = go.Figure()
<<<<<<< HEAD:experiments/mnist/eval_mnist_vary_HN.py
    hessian_fig.add_trace(go.Scatter(x=args.hidden_nodes, y=list(hessian_dims.values()), mode='markers'))
=======
    hessian_fig.add_trace(go.Scatter(x=hidden_conv_layers, y=list(hessian_dims.values()), mode='markers', name='Raw'))
    hessian_fig.add_trace(go.Scatter(x=hidden_conv_layers, y=list(hessian_dims_norm.values()), mode='markers', name='Normalised'))
>>>>>>> 8dae476ab524a31dd5e5b5b4bbea17a2c7c654a2:experiments/mnist/eval_mnist_vary_architecture.py
    hessian_fig.update_layout(
        title="Hessian dimensionality over models",
        xaxis_title="Hidden conv layers",
        yaxis_title="Dimensionality",
    )
    figs.append(hessian_fig)

    """
    ### LLC ESTIMATIONS FOR EACH ARCHITECTURE (Hidden nodes) AT CONVERGENCE ###
    llc_estimator = OnlineLLCEstimator(args.num_chains,                                       
                                       args.num_draws, 
                                       len(train_loader.dataset), 
                                       device=device)
    rlct_estimates = {}
    rlct_estimates_norm = {}
    neg_log_likelyhoods = {}
    for title, model in models.items():
        results = run_callbacks(train_loader,
                                model=model,
                                args=args,
                                callbacks=[llc_estimator],
                                criterion=criterion["general"],
                                device=device)
        #rlct_estimates_norm.append(results["llc/means"][-1]/count_parameters(model))
        rlct_estimates[title] = results["llc/means"][-1]
        rlct_estimates_norm[title] = results["llc/means"][-1]/count_parameters(model)
        neg_log_likelyhoods[title] = results["loss/trace"][-1][-1] # shape of results["loss/trace"] = (1,2000)

    rlct_fig = go.Figure()
<<<<<<< HEAD:experiments/mnist/eval_mnist_vary_HN.py
    Y = [rlct_estimates[f"{hn} HN {args.hidden_layers} HL"] for hn in args.hidden_nodes]
    #rlct_fig.add_trace(go.Scatter(x=args.hidden_nodes, y=Y, mode='markers'))
    rlct_fig.add_trace(go.Scatter(x=args.hidden_nodes, y=Y))
=======
    Y = [rlct_estimates[f"{hn} HN {out_channels} HL"] for hn in kernel_sizes]
    #rlct_fig.add_trace(go.Scatter(x=hidden_nodes, y=Y, mode='markers'))
    rlct_fig.add_trace(go.Scatter(x=kernel_sizes, y=Y))
>>>>>>> 8dae476ab524a31dd5e5b5b4bbea17a2c7c654a2:experiments/mnist/eval_mnist_vary_architecture.py
    rlct_fig.update_layout(
        title=f"Adam RLCT estimation, Elasticity : {args.elasticity}, Noise Level : {args.noise_level}",
        xaxis_title="Hidden neurons in each layer",
        yaxis_title="RLCT",
    )
    figs.append(rlct_fig)

    rlct_fig_norm = go.Figure()
<<<<<<< HEAD:experiments/mnist/eval_mnist_vary_HN.py
    Y_norm = [rlct_estimates_norm[f"{hn} HN {args.hidden_layers} HL"] for hn in args.hidden_nodes]
    rlct_fig_norm.add_trace(go.Scatter(x=args.hidden_nodes, y=Y_norm))
=======
    Y_norm = [rlct_estimates_norm[f"{hn} HN {out_channels} HL"] for hn in kernel_sizes]
    rlct_fig_norm.add_trace(go.Scatter(x=kernel_sizes, y=Y_norm))
>>>>>>> 8dae476ab524a31dd5e5b5b4bbea17a2c7c654a2:experiments/mnist/eval_mnist_vary_architecture.py
    rlct_fig_norm.update_layout(
        title=f"Adam RLCT_norm estimation, Elasticity : {args.elasticity}, Noise Level : {args.noise_level}",
        xaxis_title="Hidden neurons in each layer",
        yaxis_title="RLCT_norm",
    )
    figs.append(rlct_fig_norm)
    """

    ### VISUALISE TRAINING / TESTING LOSS OVER MODEL ARCHITECTURES ###
    train_test_fig = go.Figure()
    train_test_fig.add_trace(go.Bar(
        x=titles,
        y=[model_data["train_losses"][-1] for model_data in models_data],
        name="Training Losses",
        marker_color="indianred",
    ))
    train_test_fig.add_trace(go.Bar(
        x=titles,
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

    """
    ### VISUALISE GENERALISATION LOSS OVER MODEL ARCHITECTURES (Hidden Nodes) ###
    generalisation_losses = {}
    for model_data in models_data:
        HN, HL = model_data["description"]["LMHN"], model_data["description"]["LMHL"]
        title = f"{HN} HN {HL} HL"
        #generalisation loss equation
        generalisation_losses[title] = neg_log_likelyhoods[title] - rlct_estimates[title]/args.num_draws
    generalisation_fig = go.Figure()
    generalisation_fig.add_trace(go.Scatter(
        x=titles,
        y=[generalisation_losses[title] for title in titles],
        name="Generalisation Losses",
        marker_color="indianred",
    ))
    generalisation_fig.update_layout(
        title="Generalisation losses of model architectures",
        xaxis_title="Model",
        yaxis_title="Generalisation Loss",
        #barmode="group",
    )
    figs.append(generalisation_fig)
    """

    ### PUSH FIGURES TO LOCAL HTML FILE ###
    curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    write_figs_to_html(figs, f"./experiments/mnist/figs/mnist_hidden_nodes_{curr_time}.html", title="Investigating effect of hidden conv layers on CNN Hessian eigenspectrum")

if __name__ == "__main__":
    freeze_support()    # ONLY REQUIRED FOR WINDOWS, REMOVE IF USING MAC OR LINUX
    parser = get_mnist_vary_HN_args_parser()
    args = parser.parse_args()
    main(args)
