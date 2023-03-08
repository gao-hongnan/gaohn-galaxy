# Generalized Linear Models

Extracted from {cite}`pml1Book`:

In earlier chapters , we discussed logistic regression, which, in the binary case, corresponds to the model $\mathbb{P}(y \mid \boldsymbol{x}, \boldsymbol{w})=\operatorname{Ber}\left(y \mid \sigma\left(\boldsymbol{w}^{\top} \boldsymbol{x}\right)\right)$. In Chapter 11, we discussed linear regression, which corresponds to the model $\mathbb{P}(y \mid \boldsymbol{x}, \boldsymbol{w})=\mathcal{N}\left(y \mid \boldsymbol{w}^{\top} \boldsymbol{x}, \sigma^2\right)$. These are obviously very similar to each other. In particular, the mean of the output, $\mathbb{E}[y \mid \boldsymbol{x}, \boldsymbol{w}]$, is a linear function of the inputs $\boldsymbol{x}$ in both cases.

It turns out that there is a broad family of models with this property, known as generalized linear models or GLMs [MN89].

A GLM is a conditional version of an exponential family distribution (Section 3.4), in which the natural parameters are a linear function of the input. More precisely, the model has the following form:

$$
p\left(y_n \mid \boldsymbol{x}_n, \boldsymbol{w}, \sigma^2\right)=\exp \left[\frac{y_n \eta_n-A\left(\eta_n\right)}{\sigma^2}+\log h\left(y_n, \sigma^2\right)\right]
$$

where $\eta_n \triangleq \boldsymbol{w}^{\top} \boldsymbol{x}_n$ is the (input dependent) natural parameter, $A\left(\eta_n\right)$ is the log normalizer, $\mathcal{T}(y)=y$ is the sufficient statistic, and $\sigma^2$ is the dispersion term. ${ }^1$

We will denote the mapping from the linear inputs to the mean of the output using $\mu_n=\ell^{-1}\left(\eta_n\right)$, where the function $\ell$ is known as the link function, and $\ell^{-1}$ is known as the mean function.

Based on the results in Section 3.4.3, we can show that the mean and variance of the response variable are as follows:

$$
\begin{aligned}
\mathbb{E}\left[y_n \mid \boldsymbol{x}_n, \boldsymbol{w}, \sigma^2\right] & =A^{\prime}\left(\eta_n\right) \triangleq \ell^{-1}\left(\eta_n\right) \\
\mathbb{V}\left[y_n \mid \boldsymbol{x}_n, \boldsymbol{w}, \sigma^2\right] & =A^{\prime \prime}\left(\eta_n\right) \sigma^2
\end{aligned}
$$