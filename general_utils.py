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

from devinterp.slt import estimate_learning_coeff
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from devinterp.slt import sample
from devinterp.slt.llc import OnlineLLCEstimator
from devinterp.slt.wbic import OnlineWBICEstimator

from approxngd import KFAC
from PyHessian.pyhessian import *
from PyHessian.density_plot import *

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """"
    Train one epoch of a model.
    `model`: the nn.Module to be trained,
    `train_loader`: the PyTorch DataLoader for the training data,
    `optimizer` : the optimizer class used,
    `criterion` : loss function.
    `device` : whether cuda gpu or cpu
    """
    
    model.train()
    train_loss = 0
    for image, label in tqdm(train_loader):
        image, label = image.to(device), label.to(device)

        # TODO: checks if optimizer is of type KFAC
        if isinstance(optimizer, KFAC):
            model.zero_grad()
            # Estimate with model distribution
            with optimizer.track_forward():
                output = model(image)
                loss = criterion["kfac"](output, label)
            with optimizer.track_backward():
                loss.backward()
            optimizer.update_cov()
            # Compute loss to backprop
            model.zero_grad()
            output = model(image)
            loss = criterion["kfac"](output, label)
            loss.backward()
            optimizer.step(loss=loss)
        else:
            optimizer.zero_grad()
            output = model(image)
            loss = criterion["general"](output, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
    return train_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model with testing data.
    `model` : model to test,
    `test_loader` : PyTorch DataLoader for test data,
    `criterion` : loss function.
    `device` : whether cuda gpu or cpu
    """

    model.eval()
    test_loss = 0
    with t.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion["general"](output, label)
            test_loss += loss.item()
    return test_loss / len(test_loader)

def count_parameters(model):
    """
    Given a model, return the total number of parameters.
    """

    return sum(layer.numel() for layer in model.parameters())

def find_value(data, target, tolerance=1e-9):
    """
    Given an array of shape (2, seq_length), find the eigenvalue
    at which a certain probability density is first achieved.
    """

    for x, y in zip(data["x"], data["y"]):
        if abs(y - target) < tolerance and x > 0:
            return x
    return -1 # if not found

def sample_from_distribution(eigenvalues, densities, N):
    """
    Given a probability distribution of eigenvalues, sample N
    values from this distribution.
    """

    # Ensure eigenvalues are sorted
    indices = np.argsort(eigenvalues)
    eigenvalues = np.array(eigenvalues)[indices]
    densities = np.array(densities)[indices]

    # Normalize the probability densities to make the area under the curve equal to 1
    diffs = np.diff(eigenvalues)
    areas = diffs * (densities[:-1] + densities[1:]) / 2
    total_area = np.sum(areas)
    normalized_densities = densities / total_area

    # Construct the cumulative distribution function (CDF)
    cdf = np.zeros(len(eigenvalues))
    for i in range(1, len(cdf)):
        cdf[i] = cdf[i-1] + diffs[i-1] * (normalized_densities[i-1] + normalized_densities[i]) / 2

    # Sample from the distribution
    samples = []
    for _ in range(N):
        # Generate a random number between 0 and 1
        u = np.random.rand()

        # Find the segment where the random number falls in the CDF
        idx = np.searchsorted(cdf, u) - 1
        idx = max(idx, 0)  # Ensure idx is not negative

        # Linear interpolation within the segment to find the sample
        if idx < len(cdf) - 1:
            slope = (cdf[idx + 1] - cdf[idx]) / (eigenvalues[idx + 1] - eigenvalues[idx])
            sample = eigenvalues[idx] + (u - cdf[idx]) / slope
        else:
            sample = eigenvalues[idx]

        samples.append(sample)

    return samples

def write_figs_to_html(figs, dest, title):
    """
    Given a list of Plotly figures, store them in a local HTML file.
    figs : a List of figures to export.
    dest : path to destination.
    title : HTML file title.
    """

    div_figs = [fig.to_html(full_html=False) for fig in figs]

    graphs_html = ''.join(f'<div>{div}</div>' for div in div_figs)

    html_template = """
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
            }}
        </style>
        <title>MARS experiment</title>
    </head>
    <body>
        <h1>{title}</h1>
        {graphs}
    </body>
    </html>
    """
    final_html = html_template.format(graphs=graphs_html, title=title)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(final_html)

def run_callbacks(train_loader, model, criterion, callbacks, args, device):
    """
    Perform LLC (local learning coefficient) estimation on a model.
    """
    
    optim_kwargs = {
        "lr" : 1e-6,
        "elasticity" : args.elasticity,
        "temperature" : "adaptive",
        "num_samples" : len(train_loader.dataset),
        "save_noise" : True,
    }

    sample(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer_kwargs=optim_kwargs,
        sampling_method=SGLD,
        num_chains=args.num_chains,
        num_draws=args.num_draws,
        callbacks=callbacks,
        device=device
    )

    results = {}

    for callback in callbacks:
        if hasattr(callback, "sample"):
            results.update(callback.sample())
    
    return results

def load_models(base_path, criteria):
    """
    Load models that match specific criteria with dynamic parameter parsing.

    Parameters:
    - base_path (str): Path to the directory containing the models.
    - criteria (dict): Dictionary of criteria for filtering models. Key is parameter name, value is desired value or a list of acceptable values.

    Returns:
    - List of model state dictionaries that match the criteria.
    """

    state_dicts = []
    models_data = []

    # List all files in the models directory
    for filename in os.listdir(base_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(base_path, filename)
            with open(file_path, "rb") as file:
                model = pickle.load(file)
            desc = model["args"]
            print(desc)
            print(criteria)
            if all(desc.get(key) in (value if isinstance(value, list) else [value]) for key, value in criteria.items()):
                model_data = {}
                state_dicts.append(model["state_dict"])
                model_data["description"] = model["args"]
                model_data["train_losses"] = model["train_losses"]
                model_data["test_losses"] = model["test_losses"]
                models_data.append(model_data)

    return state_dicts, models_data
    

        



    







    
