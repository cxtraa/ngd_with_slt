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
4. [Testing](#testing)
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

When using `wandb`, you will be prompted for an API key. Follow the provided instructions, and you should be able to access the team "slt_to_the_moon".

5. Using GPU **[WIP]**


## Training

The network can be trained using the `train.py` script. For training on SHTechPartA, use

```
python train.py --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0
```
By default, a periodic evaluation will be conducted on the validation set.

## Testing

A trained model (with an MAE of **51.96**) on SHTechPartA is available at "./weights", run the following commands to launch a visualization demo:

```
python run_test.py --weight_path ./weights/SHTechA.pth --output_dir ./logs/
```
## TODO

* Make PyHessian work for multiple batches of data
* Set up checkpointing, saving and loading models (to Weights and Biases)
    * currently it is to save as pickle file, whereas wandb treats it as new run upon running eval.py
    * need to use wandb run id to do this
* wandb takes very long to upload data, not sure why
* there are quite lines of repeated code in both train and eval (see the TODO lines), ideally should remove these
    * these can cause problems if train and eval are run with different params
* setup hyperparam testing


## About

This research program is being conducted as part of the MARS (Mentorship for Alignment Research Students) with the University of Cambridge. The project researchers are Moosa Saghir, Zack Liu, and Ragha Rao, with supervision from SERI MATS scholar Evan Ryan Gunter who is based in Berkeley, California. We are affiliated with the DevInterp team and use the `devinterp` library extensively to perform MCMC searches to find empirical estimates of the local learning coefficient near singularities.

If you are interested in developmental interpretability, see the [DevInterp](https://devinterp.com/) website which has resources and project ideas for getting into research. If you are interested in singular learning theory in general, see Liam Carroll's [distilling singular learning theory (DSLT) series](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) as a good starting point. If you would like to contribute to our research or you have a question, please [email me](emailto::ms3017@cam.ac.uk)!

The project code is entirely written in Python notebooks, with commentary and instructions on how to reproduce results included.
