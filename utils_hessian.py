"""
Utility functions specifically for working with PyHessian and generating data based on the Hessian.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import einops

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
from architectures.Linear import *

import plotly.express as px
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def get_esd_plot_plotly(eigenvalues, weights, plot_type, title=None, fig=None, name=None):
    """
    Produce eigenspectrum plot in Plotly, with support for overlaid plots.
    """

    density, grids = density_generate(eigenvalues, weights)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grids, y=density + 1.0e-7, mode='lines'))
    fig.update_layout(
        title=title,
        xaxis_title='Eigenvalue',
        yaxis_title='Density (Log Scale)',
        yaxis=dict(type=plot_type),
    )
    
    return fig

def produce_hessians(models, data_loader, num_batches, criterion, device, history):
    """
    Produces hessians from a dictionary of models

    Parameters:
    models (dict): each key is the title of model, and values are either models or a list of models
    history (Boolean): if true, models will be the model_histories and a list of values is returned instead of a single value
                        if false, models will be the family of model weights and a single value is returned

    Returns:
    a dictionary where each key represents a model architecture, and each value is 
    if history==True:
        the corresponding list of Hessian object (over all epochs)
    if history==False:
        the Hessian object
    """

    '''
    #code copied to properly load hessian_dataloader, batch_num giving some errors
    assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
    assert (50000 % args.hessian_batch_size == 0)
    batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

    if batch_num == 1:
        for inputs, labels in train_loader:
            hessian_dataloader = (inputs, labels)
            break
    else:
        hessian_dataloader = []
        for i, (inputs, labels) in enumerate(train_loader):
            hessian_dataloader.append((inputs, labels))
            if i == batch_num - 1:
                break

    #check batch size arg
    total_batches = len(data_loader)
    if num_batches > total_batches:
        raise ValueError(f"num_batches ({num_batches}) exceeds the total available batches ({total_batches}) in the DataLoader.")
    '''

    images, labels = [], []
    iterator = iter(data_loader)
    for _ in range(num_batches):
        image, label = next(iterator)
        images.append(image)
        labels.append(label)
    images = t.cat(images, dim=0)
    labels = t.cat(labels, dim=0)
    hessians = {}
    if history:
        for key, history in models.items():
            hessian_history=[]
            crit = criterion["kfac"] if key == "KFAC" else criterion["general"]
            for e in range(len(history)):

                hessian_history.append(hessian(history[e], crit, data=(images,labels), cuda=True if device=='cuda' else False))
                print(f"Computing hessian for {key} in epoch {e}")
            hessians[key] = hessian_history
    else:
        for key, model in models.items():
            crit = criterion["kfac"] if key == "KFAC" else criterion["general"]
            hessians[key] = hessian(model, crit, data=(images,labels), cuda=True if device=='cuda' else False)
    
    return hessians

def produce_hessian_eigenspectra(hessians, plot_type, history):
    """
    Given a Hessians for different models, return a list
    of figures containing their eigenspectra, as well as a dictionary
    containing all the trace data.

    Parameters:
    hessians (dict): the dictionary containing keys as model titles, and the hessian for that model or the list of hessians
    history (Boolean): whether hessians is a list of models, or a list of hessian_histories. decides what returns

    Returns:
    if history:
        figs(list): return a list of lists, each inner list containing eigenspectrum fig for that model and epoch
        eigenspectrum_data(dict):each key is model architecture, each value is a list of eigenspectrum data across epochs
    else:
        figs(list) return a list of figures containing their eigenspectra
        eigenspectrum_data(dict):each key is model architecture, each value is the eigenspectrum data
    """

    def get_hessian_eigenspectrum(hessian):
        '''
        for a single hessian object, return the eigenspectrum fig and trace data
        '''
        density_eigen, density_weight = hessian.density()
        #temp_fig is eigenspectrum plot for one model
        temp_fig = get_esd_plot_plotly(density_eigen, density_weight, title=f"{key} Hessian eigenspectrum", plot_type=plot_type)
        #model_figs.append(temp_fig)
        trace=temp_fig.data[0]
        trace.name=key
        eigenspectrum=({
        "x" : list(trace.x),
        "y" : list(trace.y),
        "num_params": count_parameters(hessian.model)
        })
        return temp_fig, eigenspectrum

    figs = []
    eigenspectrum_data = {}
    if history:
        # If history is True, models is a list of lists
        for key, hessian_history in hessians.items():
            #note that zip will unpack [(1,a),(2,b),(3,c)] into [(1,2,3),(a,b,c)]
            model_figs, model_eigenspectrum = zip(*[get_hessian_eigenspectrum(hessian) for hessian in hessian_history])
            figs.append(model_figs)
            eigenspectrum_data[key] = list(model_eigenspectrum)

    else:
        for key, hessian in hessians.items():
            fig, eigenspectrum = get_hessian_eigenspectrum(hessian)
            figs.append(fig)
            eigenspectrum_data[key]=eigenspectrum

    return figs, eigenspectrum_data

def produce_hessian_dimensionality(eigenspectrum_data, history):
    """
    Given a dictionary containing eigenspectrum trace data, return
    the predicted dimensionality (Hessian) for each model.

    if history is true, the values of eigespectrum_data dict are lists and hence return the dimensionality as a dict of lists
    """
    def get_hessian_dimensionality(eigenspectrum):
        '''
        get dimension for a single eigenspectrum data object
        '''
        eigenvalues = np.array(eigenspectrum["x"])
        density = np.array(eigenspectrum["y"])
        num_params = eigenspectrum["num_params"]

        cut_off = find_value(eigenspectrum, 100e-9)
        eigenvalues_small = eigenvalues[eigenvalues < cut_off]
        density_small = density[eigenvalues < cut_off]     
        area_small = simps(density_small, eigenvalues_small)
        small_eigenvalues = round(area_small * num_params)
        dimension = num_params - small_eigenvalues
        return dimension

    hessian_dims, hessian_dims_norm = {}, {}

    if history:
        # Process each model's history of eigenspectra
        for key, eigenspectra in eigenspectrum_data.items():
            dims =[get_hessian_dimensionality(es) for es in eigenspectra]
            #all the eigenspectra for each model epoch should have the same num_params
            num_params=eigenspectra[0]["num_params"]
            hessian_dims[key] = dims
            hessian_dims_norm[key] = [dim/num_params for dim in dims]
    else:
        # Process each model's single eigenspectrum
        for key, eigenspectrum in eigenspectrum_data.items():
            dimension = get_hessian_dimensionality(eigenspectrum)
            num_params=eigenspectrum["num_params"]
            hessian_dims[key] = dimension
            hessian_dims_norm[key] = dimension / num_params

        # print(f"\n====== EIGENVALUES SUMMARY: {key} ======")
        # print(f"Mean: {mu} Standard deviation: {sigma}")
        # print(f"Proportion of small eigenvalues: {area_small}")
        # print(f"Number of small eigenvalues: {small_eigenvalues} / {num_params}")
        # print(f"MODEL DIMENSONALITY ACCORDING TO HESSIAN : {dimensions}")
    
    return hessian_dims, hessian_dims_norm