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

device = "cuda" if t.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore")

#%%

def get_train_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training model', add_help=False)

    parser.add_argument('--sweep', action='store_true', help='Decide whether or not to do hyperparameter sweep. Default False.')

    #model params
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--num_epochs', default=10, type=int,help="number of epochs for training, must be at least 5 for RLCT estimate")
    parser.add_argument('--momentum', default=0.8, type=float)
    parser.add_argument('--num_draws', default=400, type=int)
    parser.add_argument('--num_chains', default=1, type=int)
    parser.add_argument('--noise_level', default=0.5, type=float)
    parser.add_argument('--elasticity', default=50, type=float)

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
    
    #%%
    #load in model
    model = NeuralNet().to(device)
    print(model)

    #%% Build dataset
    train_loader, test_loader = build_data(args)

    #finish logging
    wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_train_args_parser()])
    args = parser.parse_args()
    main(args)