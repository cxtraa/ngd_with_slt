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
from general_utils import *
from models.architectures.NN import *

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

def produce_hessians(models, data_loader, num_batches, criterion, device):
    """
    Return a dictionary where each key represents a model,
    and each value is its corresponding Hessian object from
    the PyHessian library.
    """

    images, labels = [], []
    iterator = iter(data_loader)
    for _ in range(num_batches):
        image, label = next(iterator)
        images.append(image)
        labels.append(label)
    images = t.cat(images, dim=0)
    labels = t.cat(labels, dim=0)
    hessians = {}
    for key, model in models.items():
        crit = criterion["kfac"] if key == "KFAC" else criterion["general"]
        hessians[key] = hessian(model, crit, data=(images,labels), cuda=True if device=='cuda' else False)
    
    return hessians

def produce_hessian_eigenspectra(hessians, plot_type="linear"):
    """
    Given a list of Hessian classes for different models, return a list
    of figures containing their eigenspectra, as well as a dictionary
    containing all the trace data.
    """

    overlaid_fig = go.Figure()
    figs = []
    eigenspectrum_data = {}
    for key, hessian in hessians.items():
        density_eigen, density_weight = hessian.density()
        temp_fig = get_esd_plot_plotly(density_eigen, density_weight, title=f"{key} Hessian eigenspectrum", plot_type=plot_type)
        figs.append(temp_fig)
        trace=temp_fig.data[0]
        trace.name=key
        overlaid_fig.add_trace(trace)
        eigenspectrum_data[key] = {
            "x" : list(trace.x),
            "y" : list(trace.y),
        }
        eigenspectrum_data[key]["num_params"] = count_parameters(hessian.model)
    overlaid_fig.update_layout(title="Hessian eigenspectrum of optimisers",
                    xaxis_title="Eigenvalue",
                    yaxis_title="Density",
                    legend_title="Optimisers",
                    yaxis=dict(type=plot_type),
                    )
    figs.append(overlaid_fig)
    return figs, eigenspectrum_data

def find_hessian_dimensionality(eigenspectrum_data):
    """
    Given a dictionary containing eigenspectrum trace data, return
    the predicted dimensionality (Hessian) for each model.
    """

    hessian_dims = {}
    for key, value in eigenspectrum_data.items():

        eigenvalues = np.array(value["x"])
        density = np.array(value["y"])
        num_params = eigenspectrum_data[key]["num_params"]
        cut_off = find_value(value, 100e-9)

        mu = simps(eigenvalues*density, eigenvalues)

        var = simps(eigenvalues**2 * density, eigenvalues) - mu**2
        sigma = np.sqrt(var)

        eigenvalues_small = eigenvalues[eigenvalues < cut_off]
        density_small = density[eigenvalues < cut_off]     

        area_small = simps(density_small, eigenvalues_small)
        small_eigenvalues = round(area_small * num_params)
        dimensions = num_params - small_eigenvalues
        hessian_dims[key] = dimensions / num_params

        # print(f"\n====== EIGENVALUES SUMMARY: {key} ======")
        # print(f"Mean: {mu} Standard deviation: {sigma}")
        # print(f"Proportion of small eigenvalues: {area_small}")
        # print(f"Number of small eigenvalues: {small_eigenvalues} / {num_params}")
        # print(f"MODEL DIMENSONALITY ACCORDING TO HESSIAN : {dimensions}")
    
    return hessian_dims