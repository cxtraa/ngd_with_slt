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

from devinterp.slt import estimate_learning_coeff
from devinterp.optim.sgld import SGLD
from devinterp.slt import sample
from devinterp.slt.llc import OnlineLLCEstimator
from devinterp.slt.wbic import OnlineWBICEstimator

from PyHessian.pyhessian import *
from PyHessian.density_plot import *
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector
from utils_general import *
from architectures import *

import plotly.express as px
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt

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
    Given a list of PyTorch models, return a list of Hessian objects for each model.
    """

    images, labels = [], []
    iterator = iter(data_loader)
    for _ in range(num_batches):
        image, label = next(iterator)
        images.append(image)
        labels.append(label)
    images = t.cat(images, dim=0)
    labels = t.cat(labels, dim=0)
    hessians = []
    for model in models:
        if model is None:
            hessians.append(None)
            continue
        model.eval()
        hessians.append(hessian(model, criterion, data=(images,labels), cuda=True if device=='cuda' else False))
    return hessians

def produce_hessian_traces(hessians, tol, maxIters, N):
    """
    Given a list of Hessian classes, compute their traces and return this as a list.
    """
    trace_averages = []
    for hessian in hessians:
        traces = np.array([])
        for i in range(N):
            traces = np.concatenate((traces, hessian.trace(maxIter=maxIters, tol=tol)))
        traces_mean = float(np.mean(traces))
        trace_averages.append(traces_mean)
    return trace_averages

def produce_fims(models, data_loader, device):
    """
    Produces a dictionary of FIM instances given a dictionary of models.
    models : dictionary of models
    """

    fims = []
    for model, in models:
        fim = FIM(model=model,
                  loader=data_loader,
                  representation=PMatKFAC,
                  n_output=10,
                  variant="classif_logits",
                  device=device)
        fims.append(fim)
    return fims

def produce_eigenspectra(hessians, plot_type="linear"):
    """
    Given a list of Hessian classes for different models, return a list
    of figures containing their eigenspectra, as well as a dictionary
    containing all the trace data.
    """

    eigenspectrum_data = []
    figs = []
    overlaid_fig = go.Figure()
    for i, hessian in enumerate(hessians):
        if hessian is None:
            eigenspectrum_data.append(None)
            continue
        density_eigen, density_weight = hessian.density()
        temp_fig = get_esd_plot_plotly(density_eigen, density_weight, title=f"{i} Hessian eigenspectrum", plot_type=plot_type)
        figs.append(temp_fig)
        trace=temp_fig.data[0]
        eigenspectrum_data.append({
            "x" : list(trace.x),
            "y" : list(trace.y),
        })
        eigenspectrum_data[i]["num_params"] = count_parameters(hessian.model)
        overlaid_fig.add_trace(trace)
    overlaid_fig.update_layout(title="Combined eigenspectra",
                    xaxis_title="Eigenvalue",
                    yaxis_title="Density",
                    yaxis=dict(type=plot_type),
                    )
    figs.append(overlaid_fig)
    return figs, eigenspectrum_data

def produce_fim_figs(fims, num_bins=1000):
    """
    Given a list of FIM instances, produce histograms of the diagonal elements of the FIMs.
    Returns a dictionary of Plotly figures.
    """

    overlaid_fig = go.Figure()
    fim_figs = []
    for i, fim in enumerate(fims):
        elements = fim.get_diag().cpu().numpy()
        fim_fig = go.Figure(
            data=[go.Histogram(x=elements, opacity=0.5, nbinsx=num_bins)]
        )
        fim_fig.update_layout(
            title_text=f"{i} FIM diagonal elements",
            xaxis_title_text="Value",
            yaxis_title_text="Frequency (log scale)",
            yaxis_type="log",
        )
        fim_figs.append(fim_fig)
        trace = fim_fig.data[0]
        trace.name = f"{i}"
        overlaid_fig.add_trace(trace)
    overlaid_fig.update_layout(title="Combined FIM histograms",
                    xaxis_title="Eigenvalue",
                    yaxis_title="Density",
                    legend_title="Optimisers",
                    yaxis_type="log",
                    )
    fim_figs.append(overlaid_fig)
    return fim_figs

def find_hessian_dimensionality(eigenspectrum_data):
    """
    Given a list containing eigenspectrum trace data, return
    the predicted dimensionality (Hessian) for each model.
    """

    hessian_dims = []
    for spectrum in eigenspectrum_data:
        if spectrum is None:
            hessian_dims.append(None)
            continue

        eigenvalues = np.array(spectrum["x"])
        density = np.array(spectrum["y"])
        num_params = spectrum["num_params"]
        cut_off = find_value(spectrum, 100e-9)

        eigenvalues_large = eigenvalues[eigenvalues > cut_off]
        density_large = density[eigenvalues > cut_off]     

        area_large = simps(density_large, eigenvalues_large)
        dimensions = round(area_large * num_params)
        hessian_dims.append(dimensions)
    
    return hessian_dims