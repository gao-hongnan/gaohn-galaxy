# Mixture Models

This section talks about mixture models.

## Introduction

One way to create more complex probability models is to take a convex combination of simple distributions. This is called a mixture model. This has the form

$$
\mathbb{P}(\boldsymbol{Y} = \boldsymbol{y} \mid \boldsymbol{\theta})=\sum_{k=1}^K \pi_k p_k(\boldsymbol{y})
$$

where $p_k$ is the $k^{\prime}$ th mixture component, and $\pi_k$ are the mixture weights which satisfy $0 \leq \pi_k \leq 1$ and $\sum_{k=1}^K \pi_k=1$

We can re-express this model as a hierarchical model, in which we introduce the discrete latent variable $z \in\{1, \ldots, K\}$, which specifies which distribution to use for generating the output $\boldsymbol{y}$. The prior on this latent variable is $p(z=k \mid \boldsymbol{\theta})=\pi_k$, and the conditional is $p(\boldsymbol{y} \mid z=k, \boldsymbol{\theta})=p_k(\boldsymbol{y})=p\left(\boldsymbol{y} \mid \boldsymbol{\theta}_k\right)$. That is, we define the following joint model:
$$
\begin{aligned}
p(z \mid \boldsymbol{\theta}) & =\operatorname{Cat}(z \mid \boldsymbol{\pi}) \\
p(\boldsymbol{y} \mid z=k, \boldsymbol{\theta}) & =p\left(\boldsymbol{y} \mid \boldsymbol{\theta}_k\right)
\end{aligned}
$$
where $\boldsymbol{\theta}=\left(\pi_1, \ldots, \pi_K, \boldsymbol{\theta}_1, \ldots, \boldsymbol{\theta}_K\right)$ are all the model parameters. The "generative story" for the data is that we first sample a specific component $z$, and then we generate the observations $\boldsymbol{y}$ using the parameters chosen according to the value of $z$. By marginalizing out $z$, we recover Equation (3.94):
$$
p(\boldsymbol{y} \mid \boldsymbol{\theta})=\sum_{k=1}^K p(z=k \mid \boldsymbol{\theta}) p(\boldsymbol{y} \mid z=k, \boldsymbol{\theta})=\sum_{k=1}^K \pi_k p\left(\boldsymbol{y} \mid \boldsymbol{\theta}_k\right)
$$
We can create different kinds of mixture model by varying the base distribution $p_k$, as we illustrate below.
3.5.1 Gaussian mixture models
A Gaussian mixture model or GMM, also called a mixture of Gaussians (MoG), is defined as follows:
$$
p(\boldsymbol{y} \mid \boldsymbol{\theta})=\sum_{k=1}^K \pi_k \mathcal{N}\left(\boldsymbol{y} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\right)
$$

## References and Further Readings

- Murphy, Kevin P. "Chapter 3.5. Mixture Models." In Probabilistic Machine Learning: An Introduction. MIT Press, 2022.
- Bishop, Christopher M. "Chapter 2.3.9. Mixture of Gaussians." In Pattern Recognition and Machine Learning. New York: Springer-Verlag, 2016.