"""
Customisable training script for different models.
"""

### IMPORT LIBRARIES ###
from multiprocessing import freeze_support

import os
import sys
import pickle
import pprint
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

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
from data.build_data import build_data

import plotly.express as px
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def get_train_args_parser():
    parser = argparse.ArgumentParser(description='Set parameters for training model', add_help=False)

    # Arguments organized by group
    arg_groups = {
        'Model Architecture': [
            {'name' : '--model', 'default' : 'LM', 'type' : str},   # Linear MNIST
            {'name' : '--LMHN', 'default' : 16, 'type' : int},  # Linear MNIST Hidden Nodes
            {'name' : '--LMHL', 'default' : 2, 'type' : int},   # Linear MNIST Hidden Layers
        ],
        'Training Hyperparameters': [
            {'name': '--lr', 'default': 1e-5, 'type': float},
            {'name': '--num_epochs', 'default': 5, 'type': int},
            {'name': '--optimiser', 'default': 'adam', 'type': str}
        ],
        'Data Loading Parameters': [
            {'name': '--batch_size', 'default': 128, 'type': int},
            {'name': '--num_workers', 'default': 12, 'type': int}
        ],
    }

    # Loop through the argument groups and add them to the parser
    for group_name, args in arg_groups.items():
        group = parser.add_argument_group(group_name)
        for arg in args:
            group.add_argument(arg['name'], default=arg['default'], type=arg['type'], help=arg.get('help', ''))

    return parser


def main(args):

    ### SETUP AND DATA LOADING ###
    device = "cuda" if t.cuda.is_available() else "cpu"
    filename = f"{args.model}-model_{args.optimiser}-optimiser_{args.lr}-LR_{args.num_epochs}-epochs_{args.batch_size}-batchsize"
    if args.model == "LM":
        filename += f"_{args.LMHL}-HL_{args.LMHN}-HN"
        model = LinearMNIST(hidden_layers=args.LMHL, hidden_nodes=args.LMHN).to(device)
        train_loader, test_loader = build_data(args)

    else:
        raise NotImplementedError("The requested model does not exist.")

    ### OPTIMISER AND LOSS FUNCTION ###
    criterion = nn.CrossEntropyLoss(reduction='mean') if args.optimiser=="ngd" else nn.CrossEntropyLoss()
    sgd = t.optim.SGD(model.parameters(), lr=args.lr)
    adam = t.optim.Adam(model.parameters(), lr=args.lr)
    rmsprop = t.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.8)
    ngd = KFAC(model, args.lr, 1e-3, momentum_type='regular', momentum=0.8, adapt_damping=False, update_cov_manually=True)
    optimisers = {
        'sgd' : sgd,
        'adam' : adam,
        'rmsprop' : rmsprop,
        'ngd' : ngd,
    }
    optimiser = optimisers[args.optimiser]

    ### TRAINING LOOP ###
    train_losses = []
    test_losses = []
    print(f"\n======================== Training with {args.optimiser} ==========================")
    pprint.pprint(vars(args))
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{args.num_epochs}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")

    ### EXPORTING DATA ###
    data_to_save = {
        #args becomes a dictionary
        "args" : vars(args),
        "train_losses" : train_losses,
        "test_losses" : test_losses,
        "state_dict" : model.state_dict(),
        "total_parameters": count_parameters(model)
    }
    with open(f"weights/{filename}.pkl", 'wb') as file:
        pickle.dump(data_to_save, file, protocol=pickle.HIGHEST_PROTOCOL)
        #t.save(data_to_save, file)


if __name__ == '__main__':
    parser = get_train_args_parser()
    args = parser.parse_args()
    main(args)