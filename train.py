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

device = "cuda" if t.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore")

#%%

def get_train_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training model', add_help=False)

    parser.add_argument('--sweep', action='store_true', help='Decide whether or not to do hyperparameter sweep. Default False.')

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
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
    
    #load in model
    model = NeuralNet().to(device)
    print(model)

    #finish logging
    wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_train_args_parser()])
    args = parser.parse_args()
    main(args)