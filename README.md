# Natural Gradient Descent: an SLT perspective
A research project in developmental interpretability aiming to test a possible claim of *singular learning theory (SLT)* which states that natural gradient descent (NGD) generalises worse than stochastic gradient descent (SGD).

Specifically, we aim to test the following claims:
- Do models trained with NGD converge to points with a higher RLCT (real log canonical threshold) and therefore higher complexity than the same model trained with SGD?
- Does reduced complexity of these models correlate with better generalisation loss?

We will use the following metrics to evaluate model complexity:
- RLCT (aka local learning coefficient) which is proposed by SLT to be the correct way of defining model dimensionality at a point, taking into account free (redudndant) dimensions
- $\text{rank}(H(w))$ where $H(w)$ is the Hessian at a point $w$ in parameter space $W$. This is a non-SLT measure of complexity, and we use it to provide a non-SLT perspective on the results.

To evaluate the generalisability of models trained with NGD vs. SGD, we will use:
- Test loss (non-SLT metric)
- Generalisation loss (SLT metric)

## Table of Contents

1. [Methodology](#methodology)
2. [Set-up](#set-up)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [TODO](#todo)
5. [About](#about)

## Methodology

We use a very simple network architecture:
*   128 neuron hidden layer
*   ReLU activation
*   Dropout with 0.5 probability
*   10 neuron output layer
*   Return log softmax probabilities

In future notebooks, we will also train small transformer models and CNNs. But for now we'll stick to a simple example to get started.

For different model architectures, we train the model using various gradient descent algorithms, including: SGD, Momentum, Adam, and NGD. We then calculate the Hessian and the RLCT at the minima converged to. This will allow comparison of the model complexities achieved by different gradient descent algorithms.

## Set-up

### Linux / MacOS

1. In the terminal, clone this repo and change to its directory

```bash
git clone repository_url
cd repository_name
```

2. Create a virtual environment named `slt` and activate it

```bash
python3.10 -m venv slt
source slt/bin/activate
```

3. Install packages:

```bash
pip install -r requirements.txt
```

If some of them don't install, then try running `pip install` in the terminal. If this still doesn't work, then run `!pip install <module_name>` in the notebook itself. Note that `approxngd` is a custom package for this project

4. Using `wandb` **[WIP]**

Store your Project name, Team Name and API key in `config.json`. Ideally, `config.json` should have dummy variables when you `git push` or `git pull`, and you should key in them yourself.

5. Using GPU **[WIP]**

Refer to David's slack channel in announcements. Note for Linux/MacOS - this should be relatively straightforward. The .ssh file should be in root folder, not your project folder.

After connecting to SSH, Zihe has already git cloned the repo into zihe/ngd_with_slt. Simply run the python commands from the terminal.

## Training

The network can be trained using the `train.py` script. Use

```
python train.py --num_epochs 5
```

or any other value for the number of epochs you wish to train. Further arguments to argparse are in `train.py`. By default, a periodic evaluation will be conducted on the validation set.

## Evaluation

A trained model is available at `./models/model.pkl`, run the following commands to evaluate RLCT and Hessian measurements on it

```bash
python eval.py --num_epochs 5
```

For now we have to manually make sure num_epochs is the same as that for the trained family of models but in the future this should not be an argument in `eval.py `

## TODO

* Make PyHessian work for multiple batches of data
* Set up checkpointing, saving and loading models (to Weights and Biases)
    * currently it is to save as pickle file, whereas wandb treats it as new run upon running eval.py
    * need to use wandb run id to do this
* wandb takes very long to upload data, not sure why. ** I ACTIVATED WANDB TO OFFLINE IN THESE CODE **
* there are quite lines of repeated code in both train and eval (see the TODO lines), ideally should remove these
    * these can cause problems if train and eval are run with different params
* setup hyperparam testing
* Note that I accidentally deleted the PyHessian module, - still running into issues after cloning repo and needing to run `git submodule init` and `git submodule update`. Also I moved `get_esd_plotly` to the engine folder, I couldnt push my changes to the submodule


## About

This research program is being conducted as part of the MARS (Mentorship for Alignment Research Students) with the University of Cambridge. The project researchers are Moosa Saghir, Zack Liu, and Ragha Rao, with supervision from SERI MATS scholar Evan Ryan Gunter who is based in Berkeley, California. We are affiliated with the DevInterp team and use the `devinterp` library extensively to perform MCMC searches to find empirical estimates of the local learning coefficient near singularities.

If you are interested in developmental interpretability, see the [DevInterp](https://devinterp.com/) website which has resources and project ideas for getting into research. If you are interested in singular learning theory in general, see Liam Carroll's [distilling singular learning theory (DSLT) series](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) as a good starting point. If you would like to contribute to our research or you have a question, please [email me](emailto::ms3017@cam.ac.uk)!

The project code is entirely written in Python notebooks, with commentary and instructions on how to reproduce results included.
