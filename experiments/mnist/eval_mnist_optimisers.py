"""
EXPERIMENT NAME: Investigating the eigenspectra, Hessian dimensionality, and RLCT values converged to by different optimisers.
EXPERIMENT DESCRIPTION: We train deep neural networks on the MNIST dataset, and compare their eigenspectra and RLCT for different optimisers.

You need to change the Selection Criteria arg group yourself!
When loading models, this will be a history of models across all epochs
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
import copy

from approxngd import KFAC
from PyHessian.pyhessian import *
from PyHessian.density_plot import *
from utils_general import *
from utils_hessian import *
from utils_rlct import *
from data.build_data import build_data

import plotly as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def get_eval_mnist_optimisers_args_parser():
    parser = argparse.ArgumentParser(description='Set parameters for training model', add_help=False)

    # Arguments organized by group
    arg_groups = {
        'RLCT Hyperparameters': [
            {'name': '--num_draws', 'default': 1000, 'type': int},
            {'name': '--num_chains', 'default': 2, 'type': int},
            {'name': '--epsilon', 'default': 1e-5, 'type':float,'help':'This is the LLC step size'},
            {'name': '--gamma', 'default': 100, 'type':float,'help':'This is localization factor'},
            {'name': '--freq', 'default': 5, 'type':int ,'help':'calculate rlct every x epochs'},
        ],
        'Hessian Parameters': [
            {'name': '--hessian_batch_size', 'default': 12, 'type': int},
        ],
        'Data Loading Parameters': [
            {'name': '--batch_size', 'default': 1024, 'type': int},
            {'name': '--num_workers', 'default': 32, 'type': int},

        ],
        'Selection Criteria':[
            #change your selection criteria here
            {'name': '--criteria', 
             'default':  json.dumps({
                 'model':'CM',
                 'optimiser':['sgd','ngd'],
                 #List of available hidden nodes
                 'CMKS':2,
                 #fixed number of hidden layers
                 'CMHL':4,
                 'num_epochs':30
             }), 
             'type': json.loads,
             'help': 'Selection criteria for the model in JSON format'},
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
    metric = {"general":nn.CrossEntropyLoss(),"kfac": nn.CrossEntropyLoss(reduction='mean')}
    train_loader, test_loader = build_data(args)

    ### LOAD MODELS FROM LOCAL FILES ###
    state_dicts, models_data = load_models("./weights", criteria=args.criteria)
    assert (len(state_dicts)>1), "Check if your criteria is correct!"

    #num_epochs should be the same for each optim
    num_epochs = models_data[0]["description"]["num_epochs"]
    #epochs will include 0, which is before model starts training
    #dev_interp_epochs is the x values for rlct and gen_loss, since we only calculate them every so often
    devinterp_epochs = list(range(0, num_epochs + 1, args.freq))
    #since we have train/test data for every single epoch
    train_test_epochs= list(range(0, num_epochs + 1, 1))

    for i in range(len(state_dicts)):
        optim = models_data[i]["description"]["optimiser"]

        history=[]
        #should include the initial epoch 0 too
        for e in range(0, num_epochs+1):
            #this function returns an object of type IncompatibleKeys, make sure to append the model not the output of this!
            name, model = create_architecture(args.criteria, device)
            model.load_state_dict(state_dicts[i][e])
            #change it to eval mode
            model.eval()
            history.append(model)
        models[optim]=history

    '''
    #get_hessian currently WIP with the batch_size arg
    ### COMPUTE MODEL EIGENSPECTRA ###
    hessians = produce_hessians(models=models,
                                data_loader=train_loader,
                                num_batches=args.hessian_batch_size,
                                criterion=metric,
                                device=device,
                                history=True)
    
    ### COMPUTE FIGURES AND EIGENSPECTRUM DATA ###

    figs , eigenspectrum_data = produce_hessian_eigenspectra(hessians, plot_type="log",history=True)
    #only take last epoch for each optim, otherwise too many. this will then be a list of last-epoch figs
    figs=[inner_list[-1] for inner_list in figs]
    # Modifying the title of each Plotly object
    for fig in figs:
         fig.layout.title.text = fig.layout.title.text + " - last epoch"
    
    ### CALCULATE ESTIMATE OF NUMBER OF LARGE EIGENVALUES (DIMENSIONS) IN SPECTRUM ###
    hessian_dims, hessian_dims_norm = produce_hessian_dimensionality(eigenspectrum_data,history=True)
    hessian_dims_fig = go.Figure()

    for optimiser, dimensions in hessian_dims.items():
        hessian_dims_fig.add_trace(go.Scatter(
            x=epochs,  # Assuming this aligns with 'dimensions' length
            y=dimensions,
            mode='lines+markers',
            name=f"{optimiser}"
        ))

    hessian_dims_fig.update_layout(
        title="Hessian dimensionality overtime for different optimiser",
        xaxis_title="Epoch",
        yaxis_title="Hessian dimensions",
    )
    figs.append(hessian_dims_fig)
    '''
    #reset figs here
    figs=[]
    optimizer_colors = {
    'sgd': 'royalblue',   # Example colors, change as needed
    'ngd': 'tomato',
    }

    ### LLC ESTIMATIONS FOR EACH ARCHITECTURE AT CONVERGENCE ###
    rlct_estimates, rlct_estimates_norm, rlct_draws, neg_log_likelyhoods, rlct_moving_average = produce_rlct(models, train_loader,metric, device, args, history =True)


    rlct_fig = go.Figure()
    for optim, rlct_history in rlct_estimates.items():
        rlct_fig.add_trace(go.Scatter(
            x=devinterp_epochs,
            y=rlct_history,
            mode='lines',
            name=optim  # Setting the name of the line as the key
        ))

    rlct_fig.update_layout(
        title=f"RLCT values for optimisers",
        xaxis_title="Epoch",
        yaxis_title="RLCT",
    )
    figs.append(rlct_fig)

    ### VISUALIZE RLCT SAMPLING CALCULATION FOR LAST EPOCH
    LLC_fig = go.Figure()
    for model_data in models_data:
        optim=model_data["description"]["optimiser"]
        
        LLC_fig.add_trace(go.Scatter(
            x=np.arange(1, args.num_draws+1),
            #get the rlct draw data for the last model in a list of models over all epochs
            y=rlct_draws[optim][-1],
            name=optim,
            line=dict(dash='solid',color=optimizer_colors[optim])
        ))
    LLC_fig.update_layout(title="LLC estimation evolution for model in last epoch",
                xaxis_title="Draws",
                yaxis_title="LLC",
                legend_title="Optimiser",
                )
    figs.append(LLC_fig)

    ### VISUALIZE RLCT MOVING AVG CALCULATION FOR LAST EPOCH
    LLC_fig = go.Figure()
    for model_data in models_data:
        optim=model_data["description"]["optimiser"]
        
        LLC_fig.add_trace(go.Scatter(
            x=np.arange(1, args.num_draws+1),
            #get the rlct draw data for the last model in a list of models over all epochs
            y=rlct_moving_average[optim][-1],
            name=optim,
            line=dict(dash='solid',color=optimizer_colors[optim])
        ))
    LLC_fig.update_layout(title="LLC estimation evolution for model in last epoch",
                xaxis_title="Draws",
                yaxis_title="LLC_mov_avg",
                legend_title="Optimiser",
                )
    figs.append(LLC_fig)

    ### VISUALISE TRAINING / TESTING/ GENERALIZATION LOSS OVER OPTIMISERS ###
    loss_fig = go.Figure()

    for model_data in models_data:
        optim=model_data["description"]["optimiser"]

        loss_fig.add_trace(go.Scatter(
            x=train_test_epochs,
            y=model_data["train_losses"],
            name=optim+"- Train Loss",
            line=dict(dash='dash',color=optimizer_colors[optim])
        ))
        loss_fig.add_trace(go.Scatter(
            x=train_test_epochs,
            y=model_data["test_losses"],
            name=optim+"- Test Loss",
            line=dict(dash='solid',color=optimizer_colors[optim])
        ))

        loss_fig.add_trace(go.Scatter(
            x=devinterp_epochs,
            y=np.array(neg_log_likelyhoods[optim]),
            name=optim+"- DevInterp NLL",
            line=dict(dash='dashdot',color=optimizer_colors[optim])
        ))

        '''
        loss_fig.add_trace(go.Scatter(
            x=devinterp_epochs,
            y=np.array(neg_log_likelyhoods[optim]) - np.array(rlct_estimates[optim])/args.num_draws,
            name=optim+"- Generalization Losses - num_draws",
            line=dict(dash='dot',color=optimizer_colors[optim])
        ))

        loss_fig.add_trace(go.Scatter(
            x=devinterp_epochs,
            y=np.array(neg_log_likelyhoods[optim]) - np.array(rlct_estimates[optim])/len(train_loader.dataset),
            name=optim+"- Generalization Losses - dataset size",
            line=dict(dash='dot',color=optimizer_colors[optim])
        ))
        '''
        loss_fig.add_trace(go.Scatter(
            x=devinterp_epochs,
            y=np.array(neg_log_likelyhoods[optim]) - np.array(rlct_estimates[optim])/args.batch_size,
            name=optim+"- Generalization Losses - batch size",
            line=dict(dash='dot',color=optimizer_colors[optim])
        ))



    loss_fig.update_layout(
        title="Evolution of loss over optimisers",
        xaxis_title="Epochs",
        yaxis_title="Loss",
    )
    figs.append(loss_fig)

    ### PUSH FIGURES TO LOCAL HTML FILE ###
    curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    write_figs_to_html(figs, f"./experiments/mnist/figs/mnist_optimisers_{name}_{curr_time}.html", title="Investigating effect of optimiser on RLCT / Hessian eigenspectrum")


if __name__ == "__main__":
    freeze_support()    # ONLY REQUIRED FOR WINDOWS, REMOVE IF USING MAC OR LINUX
    parser = get_eval_mnist_optimisers_args_parser()
    args = parser.parse_args()
    main(args)
