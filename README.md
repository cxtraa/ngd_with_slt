# NGD converges to less degenerate solutions than SGD

This was a research project conducted by Moosa Saghir, Ragha Rao, and Zihe Liu, under the supervision of Evan Ryan Gunter. See the published paper [on arXiV](https://arxiv.org/abs/2409.04913).

### Abstract

The number of free parameters, or \textit{dimension}, of a model is a straightforward way to measure its complexity: a model with more parameters can encode more information. However, this is not an accurate measure of complexity: models capable of memorizing their training data often generalize well despite their high dimension \cite{doubledescent}. \textit{Effective dimension} aims to more directly capture the complexity of a model by counting only the number of parameters required to represent the functionality of the model \citep{effective_dimension_evan}. Singular learning theory (SLT) proposes the learning coefficient \( \lambda \) as a more accurate measure of effective dimension \citep{Watanabe2009}.
By describing the rate of increase of the volume of the region of parameter space around a local minimum with respect to loss, $\lambda$ incorporates information from higher-order terms. We compare \( \lambda \) of models trained using natural gradient descent (NGD) and stochastic gradient descent (SGD), and find that those trained with NGD consistently have a higher effective dimension for both of our methods: the Hessian trace \( \text{Tr}(\mathbf{H}) \), and the estimate of the local learning coefficient (LLC) $\hat{\lambda}(w^*)$.

## Code structure

`./networks.py` - contains nn.Module classes for models we train on CIFAR10 / MNIST.
- Deep feedforward neural networks with ReLU activations
- Convolutional neural networks
- Minimally singular toy model

`./utils_general.py` - contains utility functions for data loading, training models, evaluating models, writing figures to HTML, calculating RLCT using `devinterp`.
`./utils_hessian_fim.py` - utility functions for working with the Hessian and Fisher Information Matrix

The following are our project notebooks, which are fully reproducible and contain commentary on the experiments ran:
- `./experiments/eval_ngd_sgd.ipynb` - the notebook used to create all the experiments in the paper.
- `./experiments/eval_min_singular_models.ipynb` - some extra investigation we did on the LLC of minimally singular models, and testing whether the theory aligns with experiment.
- `./experiments/eval_mnist_phase_transitions.ipynb` - analysing phase transitions in the Hessian eigenspectrum throughout training (not included in the paper).

Results are stored in `./experiments/...` under the respective folder name. For example, results for problem 1 are stored as HTML files containing Plotly figures in `./experiments/ngd_sgd`.

## About

This research program was conducted as part of the MARS (Mentorship for Alignment Research Students) with the University of Cambridge. We are affiliated with the growing field of developmental interpretability that aims to apply SLT to neural networks to understand how complexity in models arises through phase transitions.

If you are interested in developmental interpretability, see the [DevInterp](https://devinterp.com/) website which has resources and project ideas for getting into research. If you are interested in singular learning theory in general, see Liam Carroll's [distilling singular learning theory (DSLT) series](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) as a good starting point.

If you have any questions about our research, please do not hesitate to contact me at [ms3017@cam.ac.uk](mailto:ms3017@cam.ac.uk).
