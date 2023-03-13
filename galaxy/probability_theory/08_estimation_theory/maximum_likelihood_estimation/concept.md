# Concept

## Likelihood

### Some Intuition

***This section is adapted from {cite}`chan_2021`.***

Consider a set of $N$ data points $\mathcal{S}=\left\{x^{(1)}, x^{(2)}, \ldots, x^{(n)}\right\}$. We want to describe these data points using a probability distribution. What would be the most general way of defining such a distribution?

Since we have $N$ data points, and we do not know anything about them, the most general way to define a distribution is as a high-dimensional probability density function (PDF) $f_{\mathbf{X}}(\mathbf{x})$. This is a PDF of a random vector $\mathbf{X}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$. A particular realization of this random vector is $\mathbf{x}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$.

$f_{\mathbf{X}}(\mathbf{x})$ is the most general description for the $N$ data points because $f_{\mathbf{X}}(\mathbf{x})$ is the **joint** PDF of all variables. It provides the complete statistical description of the vector $\mathbf{X}$. For example, we can compute the mean vector $\mathbb{E}[\mathbf{X}]$, the covariance matrix $\operatorname{Cov}(\mathbf{X})$, the marginal distributions, the conditional distribution, the conditional expectations, etc. In short, if we know $f_{\mathbf{X}}(\mathbf{x})$, we know everything about $\mathbf{X}$.

The joint PDF $f_{\mathbf{X}}(\mathbf{x})$ is always **parameterized** by a certain parameter $\boldsymbol{\theta}$. For example, if we assume that $\mathbf{X}$ is drawn from a joint Gaussian distribution, then $f_{\mathbf{X}}(\mathbf{x})$ is parameterized by the mean vector $\boldsymbol{\mu}$ and the covariance matrix $\boldsymbol{\Sigma}$. So we say that the parameter $\boldsymbol{\theta}$ is $\boldsymbol{\theta}=(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. To state the dependency on the parameter explicitly, we write

$$
f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})=\mathrm{PDF} \text { of the random vector } \mathbf{X} \text { with a parameter } \boldsymbol{\theta} .
$$

When you express the joint PDF as a function of $\mathbf{x}$ and $\boldsymbol{\theta}$, you have two variables to play with. The first variable is the **observation** $\mathbf{x}$, which is given by the measured data. We usually think about the probability density function $f_{\mathbf{X}}(\mathbf{x})$ in terms of $\mathbf{x}$, because the PDF is evaluated at $\mathbf{X}=\mathbf{x}$. In estimation, however, $\mathbf{x}$ is something that you cannot control. When your boss hands a dataset to you, $\mathbf{x}$ is already fixed. You can consider the probability of getting this particular $\mathbf{x}$, but you cannot change $\mathbf{x}$.

The second variable stated in $f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$ is the **parameter** $\boldsymbol{\theta}$. This parameter is what we want to find out, and it is the subject of interest in an estimation problem. Our goal is to find the optimal $\boldsymbol{\theta}$ that can offer the "best explanation" to data $\mathbf{x}$, in the sense that it can maximize $f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$.

The likelihood function is the PDF that shifts the emphasis to $\boldsymbol{\theta}$, let's define it formally.

### Definition

```{prf:definition} Likelihood Function
:label: def:likelihood

Let $\mathbf{X}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$ be a random vector drawn from a joint PDF $f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$, and let $\mathbf{x}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$ be the realizations. The likelihood function is a
function of the parameter $\boldsymbol{\theta}$ given the realizations $\mathbf{x}$ :

$$
\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x}) \stackrel{\text { def }}{=} f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})
$$ (eq:likelihood)
```

```{prf:remark} Likelihood is not Conditional PDF
:label: rem:likelihood

A word of caution: $\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x})$ is not a conditional PDF because $\boldsymbol{\theta}$ is not a random variable. The correct way to interpret $\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x})$ is to view it as a function of $\boldsymbol{\theta}$.
```

### Independence and Identically Distributed (IID)

While $f_{\mathbf{X}}(\mathbf{x})$ provides us with a complete picture of the random vector $\mathbf{X}$, using $f_{\mathbf{X}}(\mathbf{x})$ is tedious. We need to describe how each $x^{(n)}$ is generated and describe how $x^{(n)}$ is related to $X_{m}$ for all pairs of $n$ and $m$. If the vector $\mathbf{X}$ contains $N$ entries, then there are $N^{2} / 2$ pairs of correlations we need to compute. When $N$ is large, finding $f_{\mathbf{X}}(\mathbf{x})$ would be very difficult if not impossible.

What does this mean? Two things.

1. There is no assumption of **independence** between the data points. This means that
    describing the joint PDF $f_{\mathbf{X}}(\mathbf{x})$ is very difficult.
2. Each data point *can* be drawn from a different distribution $f_{X^{(n)}}(x^{(n)})$ for
    each $n$.

Hope is not lost.

Enter the **independence and identically distributed (IID)** assumption.
This assumption states that the data points $\mathbf{x}^{(n)}$ are independent and identically distributed.

In other words, each data point $\mathbf{x}^{(n)}$ is drawn from **identical** distribution
$f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$ parameterized by $\boldsymbol{\theta}$ and
each pair of data points $\mathbf{x}^{(n)}$ and $\mathbf{x}^{(m)}$ are **independent** of each other.

Now, we can write the problem in a much simpler way, where the joint PDF $f_{\mathbf{X}}(\mathbf{x})$ is replaced by the product of the PDFs of each data point $f_{x^{(n)}}(x^{(n)})$.

$$
f_{\mathbf{X}}(\mathbf{x})=f_{x^{(1)}, \ldots, x^{(n)}}\left(x^{(1)}, \ldots, x^{(n)}\right)=\prod_{n=1}^{N} f_{x^{(n)}}\left(x^{(n)}\right) .
$$

or in our context, we can add the **parameter** $\boldsymbol{\theta}$ to the PDFs.

$$
f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})=f_{x^{(1)}, \ldots, x^{(n)}}\left(x^{(1)}, \ldots, x^{(n)}\right)=\prod_{n=1}^{N} f_{x^{(n)}}\left(x^{(n)} ; \boldsymbol{\theta}\right) .
$$

Let's formally redefine the likelihood function with the IID assumption. Note this is
an ubiquitous assumption in machine learning and therefore we will stick to this unless
otherwise stated.

```{prf:definition} Likelihood Function with IID Assumption
:label: def:likelihood-iid

Given $i.i.d.$ random variables $x^{(1)}, \ldots, x^{(n)}$ that all have the same PDF $f_{x^{(n)}}\left(x^{(n)}\right)$, the **likelihood function** is defined as:


$$
\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x}) \stackrel{\text { def }}{=} \prod_{n=1}^{N} f_{x^{(n)}}\left(x^{(n)} ; \boldsymbol{\theta}\right)
$$
```

Notice that in the previous sections, there was an implicit assumption that the random vector $\mathbf{X}$ is a vector of **univariate*** random variables. This is not always the case. In fact, most of the time, the random vector $\mathbf{X}$ is a "vector" (collection) of **multivariate** random variables in the machine
learning realm.

Let's redefine the likelihood function for the higher dimensional case, and also take the opportunity
to introduce the definition in the context of machine learning.

### Likelihood in the Context of Machine Learning

```{prf:definition} Likelihood Function with IID Assumption (Higher Dimension)
:label: def:likelihood-iid-higher-dim

Given a dataset $\mathcal{S} = \left\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\right\}$ where each $\mathbf{x}^{(n)}$ is a vector of $D$-dimensional drawn $i.i.d.$ from the same underlying distribution
$\mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right)$, the **likelihood function** is defined as:

$$
\mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) \stackrel{\text { def }}{=} \prod_{n=1}^{N} \mathbb{P}_{\mathcal{D}}\left(\mathbf{x}^{(n)} ; \boldsymbol{\theta}\right)
$$ (eq:likelihood-machine-learning)
```

```{prf:remark} Where's the $y$?
:label: rem:where-y

Some people may ask, isn't the setting in our classification problem a supervised one with labels?
Where are the $y$ in the likelihood function? Good point, the $y$ is not included in our current section
for simplicity. However, the inclusion of $y$ can be merely thought as denoting "an additional"
random variable in the likelihood function above.

For example, let's say $y$ is the target column of a classification problem on breast cancer, denoting
whether the patient has cancer or not. Then, the likelihood function can be written as:

$$
\mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) \stackrel{\text { def }}{=} \prod_{n=1}^{N} \mathbb{P}_{\mathcal{D}}\left(\mathbf{x}^{(n)}, y^{(n)} ; \boldsymbol{\theta}\right)
$$

where $\mathcal{S}$ is defined as:

$$
\mathcal{S} = \left\{\left(\mathbf{x}^{(1)}, y^{(1)}\right), \ldots, \left(\mathbf{x}^{(n)}, y^{(n)}\right)\right\}
$$
```

### The Log-Likelihood Function

We will later see in an example that why the log-likelihood function is useful. For now, let's just
say that due to numerical reasons (underflow), we will use the log-likelihood function instead of
the likelihood function. The intuition is that the likelihood defined in {eq}`eq:likelihood-machine-learning` is a product of individual PDFs. If we have 1 billion samples (i.e. $N = 1,000,000,000$), then the likelihood function will be a product of 1 billion PDFs. This is a very small number and will cause [**arithmetic underflow**](https://en.wikipedia.org/wiki/Arithmetic_underflow). The log-likelihood function is a solution to this problem.

```{prf:definition} Log-Likelihood Function
:label: def:log-likelihood

Given a dataset $\mathcal{S} = \left\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\right\}$ where each $\mathbf{x}^{(n)}$ is a vector of $D$-dimensional drawn $i.i.d.$ from the same underlying distribution
$\mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right)$, the **log-likelihood function** is defined as:

$$
\log \mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) \stackrel{\text { def }}{=} \sum_{n=1}^{N} \log \mathbb{P}_{\mathcal{D}}\left(\mathbf{x}^{(n)} ; \boldsymbol{\theta}\right)
$$ (e:log-likelihood-machine-learning)
```

One will soon see that **maximization** of the log-likelihood function is equivalent to **maximization** of the likelihood function. They give the same result.

Let's walk through an example:

```{prf:example} Log-Likelihood of Bernoulli Distribution
:label: ex:log-likelihood-bernoulli

The log-likelihood of a sequence of $i.i.d.$ Bernoulli *univariate* random variables
$x^{(1)}, \ldots, x^{(n)}$ with parameter $\theta$.

If $x^{(1)}, \ldots, x^{(n)}$ are i.i.d. Bernoulli random variables, we have

$$
f_{\mathbf{X}}(\mathbf{x} ; \theta)=\prod_{n=1}^{N}\left\{\theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}\right\} .
$$

Taking the log on both sides of the equation yields the log-likelihood function:

$$
\begin{aligned}
\log \mathcal{L}(\theta \mid \mathbf{x}) & =\log \left\{\prod_{n=1}^{N}\left\{\theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}\right\}\right\} \\
& =\sum_{n=1}^{N} \log \left\{\theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}\right\} \\
& =\sum_{n=1}^{N} x^{(n)} \log \theta+\left(1-x^{(n)}\right) \log (1-\theta) \\
& =\left(\sum_{n=1}^{N} x^{(n)}\right) \cdot \log \theta+\left(N-\sum_{n=1}^{N} x^{(n)}\right) \cdot \log (1-\theta)
\end{aligned}
$$
```

Now there will be more examples of higher-dimensional log-likelihood functions in the next section.
Furthermore, the section Maximum Likelihood Estimation for Priors in [Naive Bayes](../../machine_learning/generative/../../../machine_learning/generative/naive_bayes/concept.md) details one example of log-likelihood function for a higher-dimensional multivariate Bernoulli (Catagorical) distribution.

### Visualizing the Likelihood Function

This section mainly details how the likelihood function, despite being a function
of $\boldsymbol{\theta}$, also depends on the underlying dataset $\mathcal{S}$. The
presence of both should be kept in mind when we talk about the likelihood function.

For a more detailed analysis, see page 471-472 of Professor Stanley Chan's book "Introduction to Probability for Data Science" (see references section).

## Maximum Likelihood Estimation

After rigorously defining the likelihood function, we can now talk about the term **maximum** in maximum likelihood estimation.

The action of maximization is in itself under [optimization theory](https://en.wikipedia.org/wiki/Mathematical_optimization), a branch in mathematics. Consequently, the maximum
likelihood estimation problem is an optimization problem that seeks to find the
parameter $\boldsymbol{\theta}$ that maximizes the likelihood function.

```{prf:definition} Maximum Likelihood Estimation
:label: def:maximum-likelihood-estimation

Given a dataset $\mathcal{S}$ consisting of $N$ samples defined as:

$$
\mathcal{S} = \left\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\right\},
$$

where $\mathcal{S}$ is $i.i.d.$ generated from the distribution $\mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right)$, parametrized by $\boldsymbol{\theta}$, where the parameter $\boldsymbol{\theta}$ can be a vector of parameters defined as:

$$
\boldsymbol{\theta} = \left\{\theta_{1}, \ldots, \theta_{k}\right\}.
$$


We define the likelihood function to be:

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) \stackrel{\text { def }}{=} \mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right),
$$


then the maximum-likelihood estimate of the parameter $\boldsymbol{\theta}$ is a parameter that maximizes the likelihood function:

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}} &\stackrel{\text { def }}{=} \underset{\boldsymbol{\theta}}{\operatorname{argmax}} \mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S}\right) \\
&\stackrel{\text{ def }}{=} \mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \widehat{\boldsymbol{\theta}}\right)
\end{aligned}
$$ (eq:maximum-likelihood-estimation)
```

```{prf:remark} Maximum Likelihood Estimation for $\mathcal{S}$ with Label $y$
:label: rmk:maximum-likelihood-estimation

To be more verbose, let's also define the maximum likelihood estimate of the parameter $\boldsymbol{\theta}$ for a dataset $\mathcal{S}$ with label $y$.

First, we redefine $\mathcal{S}$ to be:

$$
\mathcal{S} = \left\{\left(\mathbf{x}^{(1)}, y^{(1)}\right), \ldots, \left(\mathbf{x}^{(n)}, y^{(n)}\right)\right\}
$$

where $\mathcal{S}$ is generated from the distribution $\mathbb{P}_{\mathcal{D}}\left(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}\right)$. The likelihood function is then defined as:

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) \stackrel{\text { def }}{=} \mathbb{P}_{\mathcal{D}}\left(\mathcal{X}, \mathcal{Y}; \boldsymbol{\theta}\right),
$$

then the maximum-likelihood estimate of the parameter $\boldsymbol{\theta}$ is a parameter that maximizes the likelihood function:

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}} &\stackrel{\text { def }}{=} \underset{\boldsymbol{\theta}}{\operatorname{argmax}} \mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S}, y\right) \\
&\stackrel{\text{ def }}{=} \mathbb{P}_{\mathcal{D}}\left(\mathcal{X}, \mathcal{Y}; \widehat{\boldsymbol{\theta}}\right)
\end{aligned}
$$
```

## References and Further Readings

- Chan, Stanley H. "Chapter 8.1. Maximum-Likelihood Estimation." In Introduction to Probability for Data Science. Ann Arbor, Michigan: Michigan Publishing Services, 2021.