# Testing SLT claims in the loss landscape
This is a research project investigating the claims of *singular learning theory (SLT)* which proposes that optimisers such as SGD favour less complex models. The model complexity is supposedly measured by the *real log canonical threshold (RLCT)*, which is a measure of the model dimensionality, and is related to traditional measures of model complexity like the Hessian rank. We aim to investigate the following research problems:

1. Are models trained using NGD more complex than those trained with SGD? (i.e. is the RLCT at convergence higher for NGD-trained models?)
2. Is it true that the RLCT is half of the Hessian rank for a minimally singular model?
3. Can we observe "phase transitions" in the evolution of the spectrum of eigenvalues of the Hessian throughout training?

We utilise the following research / tools extensively:
- the [devinterp](https://github.com/timaeus-research/devinterp/) library, used for RLCT computation
- the [PyHessian](https://github.com/amirgholami/PyHessian) library, used for approximating the Hessian eigenvalue spectrum
- the [NGD-SGD](https://github.com/YiwenShaoStephen/NGD-SGD) repository, from which `ngd.py` is obtained, for implementation of the NGD algorithm as an optimiser, as it is not supported natively in PyTorch

Our main references include:
- ['Estimating the Local Learning Coefficient at Scale'](https://arxiv.org/abs/2402.03698) by Zach Furman, and Edmund Lau
- ['PyHessian: Neural networks through the lens of the Hessian'](https://arxiv.org/abs/1912.07145) by Zhewei Yao et al.

The main findings of our research, and our methodology so far are summarised in a presentation titled `'testing_slt_claims_research_summary.pdf'.` Please refer to this presentation for an assimilation of the all the most salient graphs for our research. 

The paper for our work is still a work in progress, but expect it to be published in the next month or two.

## Code structure

`./networks.py` - contains nn.Module classes for models we train on CIFAR10 / MNIST.
- deep feedforward neural networks with ReLU activations
- convolutional neural networks
- minimally singular toy model

`./utils_general.py` - contains utility functions for data loading, training models, evaluating models, writing figures to HTML, calculating RLCT using `devinterp`.
`./utils_hessian_fim.py` - utility functions for working with the Hessian and Fisher Information Matrix

The following are our project notebooks, which are fully reproducible and contain commentary on the experiments ran:
- `./experiments/eval_ngd_sgd.ipynb` - notebook for problem 1.
- `./experiments/eval_min_singular_models.ipynb` - notebook for problem 2.
- `./experiments/eval_mnist_phase_transitions.ipynb` - notebook for problem 3.

Results are stored in `./experiments/...` under the respective folder name. For example, results for problem 1 are stored as HTML files containing Plotly figures in `./experiments/ngd_sgd`.

## About

This research program is being conducted as part of the MARS (Mentorship for Alignment Research Students) with the University of Cambridge. The project researchers are Moosa Saghir, Zack Liu, and Ragha Rao, with supervision from SERI MATS scholar Evan Ryan Gunter who is based in Berkeley, California. We are affiliated with the DevInterp team and use the `devinterp` library extensively to perform MCMC searches to find empirical estimates of the local learning coefficient near singularities.

If you are interested in developmental interpretability, see the [DevInterp](https://devinterp.com/) website which has resources and project ideas for getting into research. If you are interested in singular learning theory in general, see Liam Carroll's [distilling singular learning theory (DSLT) series](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) as a good starting point. If you would like to contribute to our research or you have a question, please [email me](emailto::ms3017@cam.ac.uk)!
