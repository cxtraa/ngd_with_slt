"""
EXPERIMENT NAME: Investigating the eigenspectra, Hessian dimensionality, and RLCT values converged to by different optimisers.
EXPERIMENT DESCRIPTION: We train deep neural networks on the MNIST dataset, and compare their eigenspectra and RLCT for different optimisers.

You need to change the Selection Criteria arg group yourself!
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
from architectures.Linear import LinearMNIST
from architectures.CNN import CnnMNIST
from data.build_data import build_data

import plotly as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def get_eval_mnist_optimisers_args_parser():
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
            #change your selection criteria here
            {'name': '--criteria', 
             'default': {
                 'model':'LM',
                 'optimiser':['sgd','ngd'],
                 #List of available hidden nodes
                 'LMHN':8,
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

    ### PRODUCE MULTIPLE MODELS FOR TRAINING WITH DIFFERENT OPTIMISERS ###

    models = {}
    for optim in args.criteria['optimiser']:

        if args.criteria['model'] == "LM":
            filename = f"LM_{args.criteria['LMHL']}-HL_{args.criteria['LMHN']}-HN"
            model = LinearMNIST(hidden_layers=args.criteria['LMHL'], hidden_nodes=args.criteria['LMHN']).to(device)
            models[optim] = model
        elif args.criteria['model'] == "CM":
            filename = f"CM_{args.criteria['CMKS']}-KS_{args.criteria['CMHL']}-HL"
            model = CnnMNIST(kernel_size=args.criteria['CMKS'], hidden_conv_layers=args.criteria['CMHL']).to(device)
        else:
            raise NotImplementedError("The requested model does not exist.")
        models[optim] = model

    criterion = {"general":nn.CrossEntropyLoss(),"kfac": nn.CrossEntropyLoss(reduction='mean')}
    train_loader, test_loader = build_data(args)

    ### LOAD MODELS FROM LOCAL FILES ###
    state_dicts, models_data = load_models("./weights", criteria=args.criteria)
    num_epochs = models_data[0]["description"]["num_epochs"]
    epochs = np.arange(1, num_epochs+1)

    for i in range(len(state_dicts)):
        optim = models_data[i]["description"]["optimiser"]   
        models[optim].load_state_dict(state_dicts[i])

    ### COMPUTE MODEL EIGENSPECTRA ###
    hessians = produce_hessians(models=models,
                                data_loader=train_loader,
                                num_batches=args.hessian_batch_size,
                                criterion=criterion,
                                device=device)
    
    ### COMPUTE FIGURES AND EIGENSPECTRUM DATA ###
    figs, eigenspectrum_data = produce_hessian_eigenspectra(hessians, plot_type="log")
    
    ### CALCULATE ESTIMATE OF NUMBER OF LARGE EIGENVALUES (DIMENSIONS) IN SPECTRUM ###
    hessian_dims, hessian_dims_norm = find_hessian_dimensionality(eigenspectrum_data)
    hessian_dims_fig = go.Figure()
    hessian_dims_fig.add_trace(go.Bar(
        x=args.criteria['optimiser'],
        y=list(hessian_dims.values()),
        name="Dims (Raw)",
    ))
    '''
    #hessian_dims_norm are way too small to be useful
    hessian_dims_fig.add_trace(go.Bar(
        x=args.criteria['optimiser'],
        y=list(hessian_dims_norm.values()),
        name="Dims (Normalised)",
    ))
    '''
    hessian_dims_fig.update_layout(
        title="Hessian dimensionality over optimisers",
        xaxis_title="Optimiser",
        yaxis_title="Hessian dimensions",
    )
    figs.append(hessian_dims_fig)

    ### VISUALISE TRAINING / TESTING/ GENERALIZATION LOSS OVER OPTIMISERS ###
    colors=iter(px.colors.qualitative.Plotly)
    loss_fig = go.Figure()

    for model_data in models_data:
        color=next(colors)
        loss_fig.add_trace(go.Scatter(
            x=epochs,
            y=model_data["train_losses"],
            name=model_data["description"]["optimiser"]+"-Training Loss",
            line=dict(dash='dash',color=color)
        ))
        loss_fig.add_trace(go.Scatter(
            x=epochs,
            y=model_data["test_losses"],
            name=model_data["description"]["optimiser"]+"-Testing Loss",
            line=dict(dash='solid',color=color)
        ))
        
        # loss_fig.add_trace(go.Bar(
        #     x=epochs,
        #     y=[neg_log_likelyhoods[title] - rlct_estimates[title]/args.num_draws for title,model in models.items()],
        #     name="Generalization Losses",
        #     marker_color="mediumseagreen",
        # ))
    loss_fig.update_layout(
        title="Evolution of loss over optimisers",
        xaxis_title="Epochs",
        yaxis_title="Loss",
    )
    figs.append(loss_fig)

    ### LLC ESTIMATIONS FOR EACH ARCHITECTURE AT CONVERGENCE ###
    rlct_estimates, rlct_estimates_norm, neg_log_likelyhoods = produce_rlct(models, train_loader,criterion, device, args)

    rlct_fig = go.Figure()
    rlct_fig.add_trace(go.Bar(
        x=list(rlct_estimates.keys()),
        y=list(rlct_estimates.values()),
        name="RLCT (Raw)"
    ))
    #These are way too small to be generally useful
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
    write_figs_to_html(figs, f"./experiments/mnist/figs/mnist_optimisers_{filename}_{curr_time}.html", title="Investigating effect of optimiser on RLCT / Hessian eigenspectrum")


if __name__ == "__main__":
    freeze_support()    # ONLY REQUIRED FOR WINDOWS, REMOVE IF USING MAC OR LINUX
    parser = get_eval_mnist_optimisers_args_parser()
    args = parser.parse_args()
    main(args)
