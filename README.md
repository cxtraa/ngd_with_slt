# NGD converges to less degenerate solutions than SGD
This research project investigates the degeneracy of Natural Gradient Descent (NGD) compared to Stochastic Gradient Descent (SGD), using Singular Learning Theory. We measure model complexity (effective dimension) using an [estimate of the local learning coefficient (LLC)](https://arxiv.org/abs/2402.03698), and find that Natural Gradient Descent leads to models with a higher LLC than SGD.

## Results

The full paper is available [here](https://arxiv.org/abs/2409.04913). A summary with relevant background and key figures is available [here](https://drive.google.com/file/d/1EAmXcjph-lRaEIwJzd8w4HxRMFNVDBs0/view?usp=sharing).

We utilise the following research tools:
- the [devinterp](https://github.com/timaeus-research/devinterp/) library, used for LLC estimation
- the [PyHessian](https://github.com/amirgholami/PyHessian) library, used for approximating the Hessian eigenvalue spectrum
- the [NGD-SGD](https://github.com/YiwenShaoStephen/NGD-SGD) repository, from which `ngd.py` is obtained, for implementation of the NGD algorithm as an optimiser, as it is not supported natively in PyTorch

## Code structure

### Notebooks

To quickly reproduce the relevant results, use the project notebooks available under the `./experiments` folder. 

- `./experiments/eval_ngd_sgd.ipynb` - the primary notebook for our paper
- `./experiments/eval_min_singular_models.ipynb` - some extra investigation we did on the LLC of minimally singular models, and testing whether the theory aligns with experiment.
- `./experiments/eval_mnist_phase_transitions.ipynb` - analysing phase transitions in the Hessian eigenspectrum throughout training (not included in the paper).

To change the type of experiments for `eval_ngd_sgd.ipynb`, optimisers, models and more, edit the config file in `args_ngd_sgd.json`
- `model_type`: `ffnn` for a fully connected neural network (with ReLU), `dlnn` for a linear neural network, `cnn` for the Lenet-5 CNN
- `experiment_type`: either `standard` for training both `ngd` and `sgd` from epoch 0, or `swap` to continue training `ngd` from `cut_off_epochs`
- `optimizer`: either `ngd`, `sgd`, or `both` to compare both optimizers

The rest of the config arguments change the behaviour of NGD and SGD, dataset used, and hyperparameters for estimating the LLC.

`./utils_general.py` contains utility functions for data loading, training models, evaluating models, writing figures to HTML, calculating RLCT using `devinterp`. `./utils_hessian_fim.py` contains utility functions for working with the Hessian and Fisher Information Matrix.

Results are mostly stored in `./experiments/ngd_sgd` under the respective folder name. We store the figures as interactive `Plotly` objects in a HTML file.

## About

This research program is conducted under the MARS (Mentorship for Alignment Research Students) program with the [Cambridge AI Safety Hub](https://www.caish.org/). The project researchers are Moosa Saghir, Zihe Liu, and Ragha Rao, with supervision from Evan Ryan Gunter.

If you are interested in developmental interpretability, see the [DevInterp](https://devinterp.com/) website which has resources and project ideas for getting into research. If you are interested in singular learning theory in general, see Liam Carroll's [distilling singular learning theory (DSLT) series](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) as a good starting point. If you have any questions, please email [Zihe](emailto::zl559@cam.ac.uk) or [Moosa](emailto::ms3017@cam.ac.uk).

