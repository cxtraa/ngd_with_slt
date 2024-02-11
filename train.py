'''
main training script used to obtain models
'''

# Import required libraries
import os
import sys
import warnings
import numpy as np
import argparse

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

#from other files
from models.NN import NeuralNet
from data.data import build_data
from engine import train_one_epoch, evaluate

warnings.filterwarnings("ignore")

#%%

def get_train_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training model', add_help=False)

    parser.add_argument('--sweep', action='store_true', help='Decide whether or not to do hyperparameter sweep. Default False.')

    #model params
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--num_epochs', default=5, type=int,help="number of epochs for training, must be at least 5 for RLCT estimate")
    parser.add_argument('--momentum', default=0.8, type=float)
    parser.add_argument('--num_draws', default=400, type=int)
    parser.add_argument('--num_chains', default=1, type=int)
    parser.add_argument('--noise_level', default=0.5, type=float)
    parser.add_argument('--elasticity', default=50, type=float)

    #Hessian arguments
    parser.add_argument(
        '--mini-hessian-batch-size',
        type=int,
        default=200,
        help='input batch size for mini-hessian batch (default: 200)')
    parser.add_argument('--hessian-batch-size',
        type=int,
        default=200,
        help='input batch size for hessian (default: 200)')

    #dataset params
    parser.add_argument('--batch_size', default=128, type=int)

    return parser

def main(args):
    # Load in config file
    with open("config.json") as f:
        config = json.load(f)

    #%% initialise wandb
    wandb.login(key=config["wandb_api_key"])
    wandb.init(project=config["project_name"],
            entity=config["team_name"],
            name="checking adam rlct convergence",
            )
    
    #%% load in model and create optimizers
    device = "cuda" if t.cuda.is_available() else "cpu"
    #move model to CUDA
    #TODO: modularize this by allow a different model structure
    model = NeuralNet().to(device)
    print(model)

    epochs = np.arange(1, args.num_epochs+1)

    criterion = {"general":nn.CrossEntropyLoss(),"kfac": nn.CrossEntropyLoss(reduction='mean')}
    sgd = t.optim.SGD(
        model.parameters(),
        lr=args.lr,
        )
    adam = t.optim.Adam(
        model.parameters(),
        lr=args.lr,
        )
    rmsprop = t.optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )
    ngd = KFAC(model, 
            args.lr, 
            1e-3,
            momentum_type='regular',
            momentum=args.momentum,
            adapt_damping=False,
            update_cov_manually=True,
            )
    
    optimizers = [adam, rmsprop, ngd]

    #%% Build dataset
    train_loader, test_loader = build_data(args)

    #%% Train Models

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
        for epoch in range(args.num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_loss = evaluate(model, test_loader, criterion, device)
            optim_train_losses.append(train_loss)
            optim_test_losses.append(test_loss)
            optim_models.append(model)
            print(f"Epoch {epoch+1}/{args.num_epochs}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")
        train_losses[name] = optim_train_losses
        test_losses[name] = optim_test_losses
        models[name] = optim_models

    # Send training and testing data to wandb
    train_fig = go.Figure()
    for optim, train_loss in train_losses.items():
        train_fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name=optim))

    train_fig.update_layout(title="Training loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    legend_title="Optimizers"
                    )
    wandb.log({"Training Losses" : train_fig})

    test_fig = go.Figure()
    for optim, test_loss in test_losses.items():
        test_fig.add_trace(go.Scatter(x=epochs, y=test_loss, mode='lines+markers', name=optim))

    test_fig.update_layout(title="Test loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    legend_title="Optimizers",
                    )
    wandb.log({"Test Losses" : test_fig})

    #%% Get Hessian at every epoch
    # For each model, compute the eigenspectrum of the Hessian (of final model) using the PyHessian library
    # value becomes from a list of models to a single model
    final_models = {key : value[-1] for key, value in models.items()}

    #dataloader object
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(train_loader):
        hessian_dataloader.append((inputs, labels))

    hessians = {}
    for key, final_model in final_models.items():
        #note KFAC is used here
        crit = criterion["kfac"] if key == "KFAC" else criterion["general"]
        hessians[key] = hessian(final_model, crit, dataloader=hessian_dataloader, cuda=True if device=='cuda' else False)

    # Output trace of Hessian for each optimiser and send results to wandb

    print("\n======================== HESSIAN TRACE SUMMARY BEGIN ==========================")
    #trail different times to get different estimates of the trace
    n_iters = 10
    for hess in hessians:
        hess_means = []
        for _ in range(n_iters):
            hessian_trace = np.mean(hessians[hess].trace())
            hess_means.append(hessian_trace)
        final_hess_mean = np.mean(hess_means)
        print(f"Trace for {hess} : {final_hess_mean}")
    print("======================== HESSIAN TRACE SUMMARY COMPLETE ==========================")


    #finish logging
    wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_train_args_parser()])
    args = parser.parse_args()
    main(args)