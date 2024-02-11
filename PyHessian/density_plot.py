#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import math
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def get_esd_plot(eigenvalues, weights, title):
    """Produce eigenspectrum plot in Matplotlib."""

    density, grids = density_generate(eigenvalues, weights)
    plt.figure(figsize=(5,3))
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)')
    plt.xlabel('Eigenvalue')
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.show()


def get_esd_plot_plotly(eigenvalues, weights, title=None, fig=None, name=None):
    """Produce eigenspectrum plot in Plotly, with support for overlaid plots."""
    
    density, grids = density_generate(eigenvalues, weights)
    
    if not fig:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grids, y=density + 1.0e-7, mode='lines'))
        fig.update_layout(
            title=title,
            xaxis_title='Eigenvalue',
            yaxis_title='Density (Log Scale)',
            yaxis=dict(type='log'),
        )
        fig.show()
        return fig
    else:
        fig.add_trace(go.Scatter(x=grids, y=density + 1.0e-7, mode='lines', name=name))

def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.01):

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)
