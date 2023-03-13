# The Machine Learning Framework

We will not dive into deep theoratical frameworks such as PAC learning, VC dimension, etc. Instead, we will do a gentle introduction using basic probability.

## The Naive Probabilistic Framework

This section details the mathematical framework that we will use throughout. It is
naive because it is not written rigorously. But for our purpose, this is sufficient
to gain a good understanding of the concepts.

[This article here](https://mostafa-samir.github.io/ml-theory-pt1/) really lays out
the framework in an intuitive manner. Please read this before proceeding. Alongside
with Alexander Jung's book, Machine Learning: The Basics, this should give anyone
a solid foundation in how to think about machine learning.

## More Formal Framework

Read [Foundations of Machine Learning](https://cs.nyu.edu/~mohri/mlbook/) for a more formal introduction to the framework.

As an example, we refer to this important paragraph:

We consider a multi-class classification problem with $c$ classes, $c \geq 1$. Let $y=$ $\{1, \ldots, c\}$ denote the output space and $\mathcal{D}$ a distribution over $\mathcal{X} \times \mathcal{y}$. The learner receives a labeled training sample $S=\left(\left(x_1, y_1\right), \ldots,\left(x_m, y_m\right)\right) \in(\mathcal{X} \times \mathcal{y})^m$ drawn i.i.d. according to $\mathcal{D}$. As in Chapter 12, we assume that, additionally, the learner has access to a feature mapping $\Phi: X \times y \rightarrow \mathbb{R}^N$ with $\mathbb{R}^N$ a normed vector space and with $\|\Phi\|_{\infty} \leq r$. We will denote by $\mathcal{H}$ a family of real-valued functions containing the component feature functions $\Phi_j$ with $j \in[N]$. Note that in the most general case, we may have $N=+\infty$. The problem consists of using the training sample $S$ to learn an accurate conditional probability $\mathrm{p}[\cdot \mid x]$, for any $x \in X$.

This is extracted from section 13.1. Learning Problem in [Foundations of Machine Learning](https://cs.nyu.edu/~mohri/mlbook/).
This treatment is important for you to appreciate the following notations.

## Further Readings

**Work in Progress** to refer to notes below.

- [Machine Learning Theory](https://mostafa-samir.github.io)
- Jung, Alexander. Machine Learning: The Basics. Springer Nature Singapore, 2023.
- Mohri, Mehryar, Rostamizadeh Afshi and Talwalkar Ameet. Foundations of Machine Learning. The MIT Press, 2018.
