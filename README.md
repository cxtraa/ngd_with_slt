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

## About

This research program is being conducted as part of the MARS (Mentorship for Alignment Research Students) with the University of Cambridge. The project researchers are Moosa Saghir, Zack Liu, and Ragha Rao, with supervision from SERI MATS scholar Evan Ryan Gunter who is based in Berkeley, California. We are affiliated with the DevInterp team and use the `devinterp` library extensively to perform MCMC searches to find empirical estimates of the local learning coefficient near singularities.

If you are interested in developmental interpretability, see the [DevInterp](https://devinterp.com/) website which has resources and project ideas for getting into research. If you are interested in singular learning theory in general, see Liam Carroll's [distilling singular learning theory (DSLT) series](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) as a good starting point. If you would like to contribute to our research or you have a question, please [email me](emailto::ms3017@cam.ac.uk)!

The project code is entirely written in Python notebooks, with commentary and instructions on how to reproduce results included.
