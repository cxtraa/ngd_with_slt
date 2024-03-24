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
from devinterp.optim import SGLD, SGNHT

from approxngd import KFAC
from PyHessian.pyhessian import *
from PyHessian.density_plot import *

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import io

def build_data_loaders(args):
    """
    Returns train_loader and test_loader for following datasets:
    MNIST, CIFAR10.
    args must specify:
    "dataset" = "mnist", "cifar10"
    "num_workers",
    "batch_size",
    """

    if args["dataset"] == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        train_loader = t.utils.data.DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"], persistent_workers=True)
        test_loader = t.utils.data.DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"], persistent_workers=True)

    elif args["dataset"] == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        train_loader = t.utils.data.DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"], persistent_workers=True)
        test_loader = t.utils.data.DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"], persistent_workers=True)

    return train_loader, test_loader

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
    total_update_norm = 0

    for image, label in tqdm(train_loader):
        image, label = image.to(device), label.to(device)
        
        before_update = [p.clone().detach() for p in model.parameters() if p.requires_grad]
        
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        update_norm = sum([(p - before).norm().item() for p, before in zip(model.parameters(), before_update) if p.requires_grad])
        total_update_norm += update_norm / len(before_update) 

    mean_train_loss = train_loss / len(train_loader)
    mean_update_norm = total_update_norm / len(train_loader)

    return mean_train_loss, mean_update_norm

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model with testing data.
    `model` : model to test.
    `test_loader` : PyTorch DataLoader for test data.
    `criterion` : loss function.
    `device` : either "cuda" or "cpu".
    """

    model.eval()
    test_loss = 0
    with t.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
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

def write_figs_to_html(figs, dest, title, summary):
    """
    Given a list of Plotly figures, store them in a local HTML file.
    figs : a List of figures to export.
    dest : path to destination.
    title : HTML file title.
    """

    div_figs = [fig.to_html(full_html=True) for fig in figs]
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
        <p>{summary}</p>
        {graphs}
    </body>
    </html>
    """
    final_html = html_template.format(graphs=graphs_html, title=title, summary=summary)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(final_html)

def estimate_rlcts(models, data_loader, criterion, device, devinterp_args):
    """
    Given a list of models, return their RLCT values, and
    return the LLC estimation history, including the loss trace.
    models : the list of models to perform analysis on.
    data_loader : the DataLoader to provide data in batches.
    criterion : loss function.
    device : CUDA or CPU.
    sampler : SGNHT or SGLD sampling methods.
    """
    rlct_estimates = []
    history = []
    if devinterp_args["sampler"] == "sgld":
        optimizer_kwargs = {
            "lr" : devinterp_args["sampler_lr"],
            "localization" : devinterp_args["localization"],
            "save_noise" : True,
        }
    elif devinterp_args["sampler"] == "sgnht":
        optimizer_kwargs = {
            "lr" : devinterp_args["sampler_lr"],
            "diffusion_factor" : devinterp_args["diffusion_factor"],
            "save_noise" : True,
        }
    for model in tqdm(models):
        results = estimate_learning_coeff_with_summary(
            model,
            data_loader,
            criterion=criterion,
            optimizer_kwargs=optimizer_kwargs,
            sampling_method=SGLD if devinterp_args["sampler"] == "sgld" else SGNHT,
            num_chains=devinterp_args["num_chains"],
            num_draws=devinterp_args["num_draws"],
            device=device,
            online=True,
        )
        chain_rlct_estimates = np.array([results["llc/moving_avg"][i][-1] for i in range(devinterp_args["num_chains"])])
        rlct_estimate = float(np.mean(chain_rlct_estimates))
        rlct_estimates.append(rlct_estimate)
        history.append(results)
        
    return rlct_estimates, history
    

        



    







    
