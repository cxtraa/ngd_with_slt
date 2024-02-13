'''
Given a family of models, existing as a dictionary
{
    optimizer1:[epoch1_model,epoch2_model],
    optimizer2:[epoch1_model,epoch2_model],
}

conducts Hessian and RLCT analysis on it
'''

import wandb
import pickle
import os

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from devinterp.slt import estimate_learning_coeff
from devinterp.optim.sgld import SGLD

from PyHessian.pyhessian import hessian # Hessian computation
from engine import get_esd_plot_plotly

from tqdm import tqdm
import json
import argparse

from data.build_data import build_data

# TODO: wandb is set offline, takes very long to upload data not sure why
# os.environ["WANDB_MODE"] = "offline"

def get_test_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training model', add_help=False)

    #TODO: repeated model params
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--num_epochs', default=5, type=int,help="number of epochs for training, must be at least 5 for RLCT estimate")
    parser.add_argument('--momentum', default=0.8, type=float)
    parser.add_argument('--num_draws', default=400, type=int)
    parser.add_argument('--num_chains', default=1, type=int)
    parser.add_argument('--noise_level', default=0.5, type=float)
    parser.add_argument('--elasticity', default=50, type=float)

    #dataset params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=128, type=int)    

    #where to load filepath
    parser.add_argument('--load_path',default='models/model.pkl',type=str, help='the file path to load back your family of models')


    return parser
    
def main(args):
    # Load in config file
    with open("config.json") as f:
        config = json.load(f)

    #%% initialise wandb
    wandb.login(key=config["wandb_api_key"])
    wandb.init(project=config["project_name"],
            entity=config["team_name"],
            name="testing adam rlct convergence",
            )
    
    #%% load in model and create optimizers
    device = "cuda" if t.cuda.is_available() else "cpu"
    #move model to CUDA
    #TODO: modularize this by allow a different model structure
    # Load the dictionary of models
    with open(args.load_path, 'rb') as file:
        models = pickle.load(file)

    # Obtain the number of epochs from the length of the model list of any optimizer
    # Using the first key in the dictionary to access its corresponding model list
    epochs = len(models[next(iter(models))])

    #%% Build dataset
    train_loader, test_loader = build_data(args)
    #TODO: these 2 lines of code are repeated, ideally there no need to define criterion again
    criterion = {"general":nn.CrossEntropyLoss(),"kfac": nn.CrossEntropyLoss(reduction='mean')}

    for optimizer, m_list in models.items():
        for m in m_list:
            m.eval()

    #%% Get Hessian at every epoch

    #TODO: make data for multiple batches, doesnt seem to work #dataloader object
    # hessian_dataloader = []
    # for i, (inputs, labels) in enumerate(train_loader):
    #     hessian_dataloader.append((inputs, labels))

    # to save time, only use one batch of training data
    for image, label in train_loader:
        break

    # For each model, compute the eigenspectrum of the Hessian (of final model) using the PyHessian library
    hessians={}
    traces={}
    n_iters = 10

    for optimizer in models:
        crit = criterion["kfac"] if optimizer == "KFAC" else criterion["general"]
        print(f"======================== Computing all Hessians and Traces for {optimizer} ==========================")
        for e, m in enumerate(models[optimizer]):
            print(f"======================== Computing all Hessians and Traces for {optimizer} and epoch {e} ==========================")
            hessian_estimate = hessian(m, crit, data=(image,label), cuda=True if device=='cuda' else False)

            model_traces = []
            for _ in range(n_iters):
                trace = np.mean(hessian_estimate.trace())
                model_traces.append(trace)
            trace_mean = np.mean(model_traces)
            print(f"Trace for {optimizer} with epoch {e} : {trace_mean}")

            if optimizer in hessians:
                hessians[optimizer].append(hessian_estimate)
                traces[optimizer].append(trace_mean)
            else:
                hessians[optimizer] = [hessian_estimate]
                traces[optimizer]=[trace_mean]


    # Produce plots of eigenspectrums for final models
    final_hessians = {key : value[-1] for key, value in hessians.items()}

    overlaid_fig = go.Figure()
    individual_figs = []

    for key, hess in final_hessians.items():
        density_eigen, density_weight = hess.density()
        temp_fig = get_esd_plot_plotly(density_eigen, density_weight, title=f"{key} Hessian eigenspectrum")
        individual_figs.append(temp_fig)

        # Assuming get_esd_plot_plotly returns a figure with one trace, add that trace to the overlaid figure
        trace=temp_fig.data[0]
        trace.name=key
        overlaid_fig.add_trace(trace)

    overlaid_fig.update_layout(title="Hessian eigenspectrum of optimisers",
                    xaxis_title="Eigenvalue",
                    yaxis_title="Density (Log Scale)",
                    legend_title="Optimisers",
                    yaxis=dict(type='log')
                    )

    #add overlaid fig to list of all figures
    individual_figs.append(overlaid_fig)

    # Log the plot data to wandb
    for fig in individual_figs:
        wandb.log({fig.layout.title.text : fig})

    #TODO: what is this line of code here for?
    wandb.log({})

    #%% evaluate RLCT for each optimizer, for each epoch
    rlct_estimates = {}
    for optimizer in models:
        print(f"======================== RLCT estimates for {optimizer} ==========================")
        for m in models[optimizer]:
            rlct_estimate = estimate_learning_coeff(
                m,
                train_loader,
                criterion=criterion["general"],
                optimizer_kwargs=dict(
                    lr=args.lr,
                    # TODO: all these args are repeated, should be taken from the model
                    noise_level=args.noise_level,
                    elasticity=args.elasticity,
                    num_samples=len(train_loader.dataset),
                    temperature="adaptive",
                ),
                sampling_method=SGLD,
                num_chains=args.num_chains,
                num_draws=args.num_draws,
                num_burnin_steps=0,
                num_steps_bw_draws=1,
                device=device,
            )
            if optimizer in rlct_estimates:
                rlct_estimates[optimizer].append(rlct_estimate)
            else:
                rlct_estimates[optimizer] = [rlct_estimate]
            print(f"RLCT estimate: {rlct_estimate}")
        average_rlct_estimate = np.sum(rlct_estimates[optimizer][-2:])/2        
        #wandb.log({"optimizer" : optimizer, "rlct_estimate" : average_rlct_estimate})
        print(f"======== FINAL RLCT ESTIMATE FOR {optimizer} : {average_rlct_estimate} ========")
    
    # Quick graphs to visualise how RLCT and Hessians evolved over time

    #plt.figure(figsize=(10, 6))
    for optim in rlct_estimates:
        #data = {"Epochs" : np.arange(1, hyperparams["num_epochs"]+1), optim : rlct_estimates[optim]}
        plt.plot(np.arange(1, args.num_epochs+1), rlct_estimates[optim], label=f'{optim} - rlct')
        plt.plot(np.arange(1, args.num_epochs+1), traces[optim], label=f'{optim} - trace')
    plt.grid()
    plt.title("RLCT + Trace vs. epochs")
    plt.xlabel("Epoch")
    plt.ylabel("RLCT or Trace")
    plt.legend()

    # Save the figure to a specified path
    plt.savefig('figs/figure.png')


if __name__ == '__main__':
    parser = get_test_args_parser()
    args = parser.parse_args()
    main(args)