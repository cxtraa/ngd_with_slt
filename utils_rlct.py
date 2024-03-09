"""
General utility functions used across our files.
"""

import os
import sys
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import einops
from scipy.integrate import simps, trapz

sys.path.append("../")

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

from tqdm import tqdm
from datetime import datetime
import json
import wandb

from devinterp.slt import estimate_learning_coeff_with_summary
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from devinterp.slt import sample
from devinterp.slt.llc import OnlineLLCEstimator
from devinterp.slt.wbic import OnlineWBICEstimator

from approxngd import KFAC
from PyHessian.pyhessian import *
from PyHessian.density_plot import *

from architectures.NN import NeuralNet
from architectures.CNN import CnnMNIST
from utils_general import count_parameters

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import io

def get_rlct_data(model,dataloader, criterion, args, device):
    """
    Perform LLC (local learning coefficient) estimation on a model.
    """
    
    optim_kwargs = {
        "lr" : args.epsilon,
        "localization":args.gamma
    }

    result = estimate_learning_coeff_with_summary(
        model=model,
        loader=dataloader,
        criterion=criterion,
        sampling_method=SGLD,
        optimizer_kwargs=optim_kwargs,
        num_draws=args.num_draws,
        num_chains=args.num_chains,
        device=device,
        verbose=True,
        online=True,
        )
    
    #take the last chain
    rlct=result["llc/means"][-1]
    rlct_norm = rlct / count_parameters(model)
    #this is shows all the rlct at each draw, averaged over chains
    rlct_draw=result["llc/means"]
    #nll = result["loss/trace"][-1][-1] #this was previous implementation, which is the last loss in the chain of the last draw
    #note that loss is of shape (num_chains,num_draws)
    #new implementation: first average across chains (sum vertically), then take the last draw
    #this is an np array
    nll = result["loss/trace"].mean(axis=0)[-1]


    return rlct, rlct_norm, rlct_draw, nll

def produce_rlct(models, dataloader,criterion, device, args,history):
    '''
    Produce RLCT data for family of models. Gets rlct every x epochs, where x is the frequency.

    Parameters:
    - models (dict): key as title, value as model
    - history(Boolean): whether or not models is only the final weights or a history containing the epoch weights

    Returns:
    - rlct_estimates (dict): keys are model titles, values are either the RLCT or a list of RLCT if a model history is passed in
    - rlct_estimates_norm (dict)
    - neg_log_likelyhoods (dict)
    - rlct_draws (dict): values are the list 
    '''
    
    rlct_estimates, rlct_estimates_norm, rlct_draws, neg_log_likelyhoods= {}, {}, {}, {}
    for title, value in models.items():

        print(len(value))

        if history:
            rlct_data=[]
            #if history contains num_epochs+1 length, then iterator goes from 0 to num_epochs, range(0,len(num_epochs+1))
            for epoch in range(0, len(value)):
                #only get rlct if fulfills frequency criteria
                if epoch% args.freq ==0:
                    #note that position zero corresponds to epoch 0
                    print(f"Calculating rlct for {title} in epoch {epoch}")
                    rlct_data.append(get_rlct_data(value[epoch],dataloader, criterion["general"], args, device))
            rlct_estimates[title], rlct_estimates_norm[title], rlct_draws[title], neg_log_likelyhoods[title] = zip(*rlct_data)
        else:
            #value is the model here
            rlct_estimates[title], rlct_estimates_norm[title], rlct_draws[title], neg_log_likelyhoods[title] = get_rlct_data(value,dataloader, criterion["general"], args, device) 

    return rlct_estimates, rlct_estimates_norm, rlct_draws, neg_log_likelyhoods



    

        



    







    
