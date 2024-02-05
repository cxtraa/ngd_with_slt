"""
Functions for working with the Hessian.
Supports:
    - Computing Hessian of model
    - Compute tridiagonalized Hessian
    - Extract eigenvalues from tridiagonalized Hessian
"""

def hessian(params):
    """
    Given model parameters and gradients, compute the Hessian matrix.
    """