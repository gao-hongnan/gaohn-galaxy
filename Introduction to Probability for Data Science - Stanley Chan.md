

Definition 8.4.
Example 8.4. Find the ML estimate for a set of i.i.d. Bernoulli random variables $\left\{x^{(1)}, \ldots, x^{(n)}\right\}$ with $x^{(n)} \sim \operatorname{Bernoulli}(\theta)$ for $n=1, \ldots, N$.

Solution. We know that the log-likelihood function of a set of i.i.d. Bernoulli random variables is given by

$$
\log \mathcal{L}(\theta \mid \mathbf{x})=\left(\sum_{n=1}^{N} x^{(n)}\right) \cdot \log \theta+\left(N-\sum_{n=1}^{N} x^{(n)}\right) \cdot \log (1-\theta)
$$

Thus, to find the ML estimate, we need to solve the optimization problem

$$
\widehat{\theta}=\underset{\theta}{\operatorname{argmax}}\left\{\left(\sum_{n=1}^{N} x^{(n)}\right) \cdot \log \theta+\left(N-\sum_{n=1}^{N} x^{(n)}\right) \cdot \log (1-\theta)\right\} .
$$

Taking the derivative with respect to $\theta$ and setting it to zero, we obtain

$$
\frac{d}{d \theta}\left\{\left(\sum_{n=1}^{N} x^{(n)}\right) \cdot \log \theta+\left(N-\sum_{n=1}^{N} x^{(n)}\right) \cdot \log (1-\theta)\right\}=0 .
$$

This gives us

$$
\frac{\left(\sum_{n=1}^{N} x^{(n)}\right)}{\theta}-\frac{N-\sum_{n=1}^{N} x^{(n)}}{1-\theta}=0
$$

Rearranging the terms yields

$$
\widehat{\theta}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}
$$

Let's do a sanity check to see if this result makes sense. The solution to this problem says that $\widehat{\theta}$ is the empirical average of the measurements. Assume that $N=50$. Let us consider two particular scenarios as illustrated in Figure $\mathbf{8 . 4}$

- Scenario 1: $\mathbf{x}$ is a vector of measurements such that $S \stackrel{\text { def }}{=} \sum_{n=1}^{N} x^{(n)}=25$. Since $N=50$, the formula tells us that $\widehat{\theta}=\frac{25}{50}=0.5$. This is the best guess based on the 50 measurements where 25 are heads. If you look at Figure 8.3 and Figure 8.4 when $S=25$, we are looking at a particular cross section in the $2 \mathrm{D}$ plot. The likelihood function we are inspecting is $\mathcal{L}(\theta \mid S=25)$. For this likelihood function, the maximum occurs at $\theta=0.5$.

\section{CHAPTER 8. ESTIMATION}
![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-490.jpg?height=694&width=1388&top_left_y=239&top_left_x=150)

Figure 8.4: Illustration of how the maximum-likelihood estimate of a set of i.i.d. Bernoulli random variables is determined. The subfigures above show two particular scenarios at $S=25$ and $S=12$, assuming that $N=50$. When $S=25$, the likelihood function has a quadratic shape centered at $\theta=0.5$. This point is also the peak of the likelihood function when $S=25$. Therefore, the ML estimate is $\widehat{\theta}=0.5$. The second case is when $S=12$. The quadratic likelihood is shifted toward the left. The $\mathrm{ML}$ estimate is $\widehat{\theta}=0.24$.

- Scenario 2: $\mathbf{x}$ is a vector of measurements such that $S \stackrel{\text { def }}{=} \sum_{n=1}^{N} x^{(n)}=12$. The formula tells us that $\widehat{\theta}=\frac{12}{50}=0.24$. This is again the best guess based on the 50 measurements where 12 are heads. Referring to Figure $\mathbf{8 . 3}$ and Figure 8.4 we can see that the likelihood function corresponds to another cross section $\mathcal{L}(\theta \mid S=12)$ where the maximum occurs at $\theta=0.24$.

At this point, you may wonder why the shape of the likelihood function $\mathcal{L}(\theta \mid \mathbf{x})$ changes so radically as $\mathbf{x}$ changes? The answer can be found in Figure 8.5. Imagine that we have $N=50$ measurements of which $S=40$ give us heads. If these i.i.d. Bernoulli random variables have a parameter $\theta=0.5$, it is quite unlikely that we will get 40 out of 50 measurements to be heads. (If it were $\theta=0.5$, we should get more or less 25 out of 50 heads.) When $S=40$, and without any additional information about the experiment, the most logical guess is that the Bernoulli random variables have a parameter $\theta=0.8$. Since the measurement $S$ can be as extreme as 0 out of 50 or 50 out of 50 , the likelihood function $\mathcal{L}(\theta \mid \mathbf{x})$ has to reflect these extreme cases. Therefore, as we change $\mathbf{x}$, we observe a big change in the shape of the likelihood function.

As you can see from Figure 8.5, $S=40$ corresponds to the marked vertical cross section. As we determine the maximum-likelihood estimate, we search among all the possibilities, such as $\theta=0.2, \theta=0.5, \theta=0.8$, etc. These possibilities correspond to the horizontal lines we drew in the figure. Among those horizontal lines, it is clear that the best estimate occurs when $\theta=0.8$, which is also the ML estimate.

\subsection{MAXIMUM-LIKELIHOOD ESTIMATION}

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-491.jpg?height=535&width=1081&top_left_y=233&top_left_x=326)

Figure 8.5: Suppose that we have a set of measurements such that $S=40$. To determine the $\mathrm{ML}$ estimate, we look at the vertical cross section at $S=40$. Among the different candidate parameters, e.g., $\theta=0.2, \theta=0.5$ and $\theta=0.8$, we pick the one that has the maximum response to the likelihood function. For $S=40$, it is more likely that the underlying parameter is $\theta=0.8$ than $\theta=0.2$ or $\theta=0.5$.

\section{Visualizing ML estimation as $N$ grows}

Maximum-likelihood estimation can also be understood directly from the PDF instead of the likelihood function. To explain this perspective, let's do a quick exercise.

Practice Exercise 8.2. Suppose that $x^{(n)}$ is a Gaussian random variable. Assume that $\sigma=1$ is known but the mean $\theta$ is unknown. Find the ML estimate of the mean.

Solution. The ML estimate $\widehat{\theta}$ is

$$
\begin{aligned}
\widehat{\theta} & =\underset{\theta}{\operatorname{argmax}} \log \mathcal{L}(\theta \mid \mathbf{x}) \\
& =\underset{\theta}{\operatorname{argmax}} \log \left\{\prod_{n=1}^{N} \frac{1}{\sqrt{2 \pi}} \exp \left\{-\frac{\left(x^{(n)}-\theta\right)^{2}}{2}\right\}\right\} \\
& =\underset{\theta}{\operatorname{argmax}}-\frac{N}{2} \log (2 \pi)-\frac{1}{2} \sum_{n=1}^{N}\left(x^{(n)}-\theta\right)^{2} .
\end{aligned}
$$

Taking the derivative with respect to $\theta$, we obtain

$$
\frac{d}{d \theta}\left\{-\frac{N}{2} \log (2 \pi)-\frac{1}{2} \sum_{n=1}^{N}\left(x^{(n)}-\theta\right)^{2}\right\}=0 .
$$

This gives us $\sum_{n=1}^{N}\left(x^{(n)}-\theta\right)=0$. Therefore, the ML estimate is

$$
\widehat{\theta}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}
$$

Now we will draw the PDF and compare it with the measured data points. Our focus

\section{CHAPTER 8. ESTIMATION}

is to analyze how the ML estimate changes as $N$ grows.

When $N=1$. There is only one observation $x^{(1)}$. The best Gaussian that fits this sample must be the one that is centered at $x^{(1)}$. In fact, the optimization is $\AA^{1}$

$$
\begin{aligned}
\widehat{\theta}=\underset{\theta}{\operatorname{argmax}} \log L\left(\theta \mid x^{(1)}\right) & =\underset{\theta}{\operatorname{argmax}} \log \left\{\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{\left(x^{(1)}-\theta\right)^{2}}{2 \sigma^{2}}\right\}\right\} \\
& =\underset{\theta}{\operatorname{argmax}}-\left(x^{(1)}-\theta\right)^{2}=x^{(1)} .
\end{aligned}
$$

Therefore, the ML estimate is $\widehat{\theta}=x^{(1)}$. Figure 8.6 illustrates this case. As we conduct the ML estimation, we imagine that there are a few candidate PDFs. The ML estimation says that among all these candidate PDFs we need to find one that can maximize the probability of obtaining the observation $x^{(1)}$. Since we only have one observation, we have no choice but to pick a Gaussian centered at $x^{(1)}$. Certainly the sample $x^{(1)}=x^{(1)}$ could be bad, and we may find a wrong Gaussian. However, with only one sample there is no way for us to make better decisions.

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-492.jpg?height=467&width=832&top_left_y=958&top_left_x=424)

Figure 8.6: $N=1$. Suppose that we are given one observed data point located around $x=-2.1$. To conduct the ML estimation we propose a few candidate PDFs, each being a Gaussian with unit variance but a different mean $\theta$. The ML estimate is a parameter $\theta$ such that the corresponding PDF matches the best with the observed data. In this example the best match happens when the estimated Gaussian PDF is centered at $x^{(1)}$.

When $N=2$. In this case we need to find a Gaussian that fits both $x^{(1)}$ and $x^{(2)}$. The probability of simultaneously observing $x^{(1)}$ and $x^{(2)}$ is determined by the joint distribution. By independence we then have

$$
\begin{aligned}
\widehat{\theta} & =\underset{\theta}{\operatorname{argmax}} \log \left\{\left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)^{2} \exp \left\{-\frac{\left.\left(x^{(1)}-\theta\right)^{2}+\left(x^{(2)}-\theta\right)^{2}\right)}{2 \sigma^{2}}\right\}\right\} \\
& =\underset{\theta}{\operatorname{argmax}}\left\{-\frac{\left(x^{(1)}-\theta\right)^{2}+\left(x^{(2)}-\theta\right)^{2}}{2 \sigma^{2}}\right\}=\frac{x^{(1)}+x^{(2)}}{2},
\end{aligned}
$$

${ }^{1}$ We skip the step of checking whether the stationary point is a maximum or a minimum, which can be done by evaluating the second-order derivative. In fact, since the function $-\left(x^{(1)}-\theta\right)^{2}$ is concave in $\theta$, a stationary point must be a maximum.

\subsection{MAXIMUM-LIKELIHOOD ESTIMATION}

where the last step is obtained by taking the derivative:

$$
\frac{d}{d \theta}\left\{\left(x^{(1)}-\theta\right)^{2}+\left(x^{(2)}-\theta\right)^{2}\right\}=2\left(x^{(1)}-\theta\right)+2\left(x^{(2)}-\theta\right) .
$$

Equating this with zero yields the solution $\theta=\frac{x^{(1)}+x^{(2)}}{2}$. Therefore, the best Gaussian that fits the observations is $\operatorname{Gaussian}\left(\frac{x^{(1)}+x^{(2)}}{2}, \sigma^{2}\right)$.

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-493.jpg?height=466&width=845&top_left_y=514&top_left_x=481)

Figure 8.7: $N=2$. Suppose that we are given two observed data points located around $x^{(1)}=-0.98$ and $x^{(2)}=-1.15$. To conduct the $\mathrm{ML}$ estimation we propose a few candidate PDFs, each being a Gaussian with unit variance but a different mean $\theta$. The ML estimate is a parameter $\theta$ such that the corresponding PDF best matches the observed data. In this example the best match happens when the estimated Gaussian PDF is centered at $\left(x^{(1)}+x^{(2)}\right) / 2 \approx-1.07$.

Does this result make sense? When you have two data points $x^{(1)}$ and $x^{(2)}$, the ML estimation is trying to find a Gaussian that can best fit both of these two data points. Your best bet here is $\widehat{\theta}=\left(x^{(1)}+x^{(2)}\right) / 2$, because there are no other choices. If you choose $\widehat{\theta}=x^{(1)}$ or $\widehat{\theta}=x^{(2)}$, it cannot be a good estimate because you are not using both data points. As shown in Figure 8.7 for these two observed data points $x^{(1)}$ and $x^{(2)}$, the PDF marked in red (which is a Gaussian centered at $\left(x^{(1)}+x^{(2)}\right) / 2$ ) is indeed the best fit.

When $N=10$ and $N=100$. We can continue the above calculation for $N=10$ and $N=100$. In this case the MLE is

$$
\begin{aligned}
\widehat{\theta} & =\underset{\theta}{\operatorname{argmax}} \log \left\{\left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)^{N} \exp \left\{-\frac{\left(x^{(1)}-\theta\right)^{2}+\cdots+\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}\right\}\right\} \\
& =\underset{\theta}{\operatorname{argmax}}-\sum_{n=1}^{N} \frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)} .
\end{aligned}
$$

where the optimization is solved by taking the derivative:

$$
\frac{d}{d \theta} \sum_{n=1}^{N}\left(x^{(n)}-\theta\right)^{2}=-2 \sum_{n=1}^{N}\left(x^{(n)}-\theta\right)
$$

Equating this with zero yields the solution $\theta=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}$.

The result suggests that for an arbitrary number of training samples the ML estimate is the sample average. These cases are illustrated in Figure 8.8. As you can see, the red curves (the estimated PDF) are always trying to fit as many data points as possible.

The above experiment tells us something about the ML estimation:

\section{CHAPTER 8. ESTIMATION}

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-494.jpg?height=381&width=672&top_left_y=237&top_left_x=172)

(c) $N=10$

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-494.jpg?height=381&width=666&top_left_y=237&top_left_x=873)

(d) $N=100$

Figure 8.8: When $N=10$ and $N=100$, the $\mathrm{ML}$ estimation continues to evaluate the different candidate PDFs. For a given set of data points, the ML estimation picks the best PDF to fit the data points. In this Gaussian example it was shown that the optimal parameter is $\widehat{\theta}=(1 / N) \sum_{n=1}^{N} x^{(n)}$, which is the sample average.

How does $\mathrm{ML}$ estimation work, intuitively?

- The likelihood function $\mathcal{L}(\theta \mid \mathbf{x})$ measures how "likely" it is that we will get $\mathbf{x}$ if the underlying parameter is $\theta$.

- In the case of a Gaussian with an unknown mean, you move around the Gaussian until you find a good fit.

\subsubsection{Application 1: Social network analysis}

ML estimation has extremely broad applicability. In this subsection and the next we discuss two real examples. We start with an example in social network analysis.

In Chapter 3, when we discussed the Bernoulli random variables, we introduced the Erdős-Rényi graph - one of the simplest models for social networks. The Erdős-Rényi graph is a single-membership network that assumes that all users belong to the same cluster. Thus the connectivity between users is specified by a single parameter, which is also the probability of the Bernoulli random variable.

In our discussions in Chapter 3 we defined an adjacency matrix to represent a graph. The adjacency matrix is a binary matrix, with the $(i, j)$ th entry indicating an edge connecting nodes $i$ and $j$. Since the presence and absence of an edge is binary and random, we may model each element of the adjacency matrix as a Bernoulli random variable

$$
X_{i j} \sim \operatorname{Bernoulli}(p)
$$

In other words, the edge $X_{i j}$ linking user $i$ and user $j$ in the network is either $X_{i j}=1$ with probability $p$, or $X_{i j}=0$ with probability $1-p$. In terms of notation, we define the matrix $\mathbf{X} \in \mathbb{R}^{N \times N}$ as the adjacency matrix, with the $(i, j)$ th element being $X_{i j}$.

A few examples of a single-membership Erdős-Rényi graph are shown in Figure 8.9. As the figure shows, the network connectivity increases as the Bernoulli parameter $p$ increases. This happens because $p$ defines the "density" of the edges. If $p$ is large, we have a greater chance of getting $X_{i j}=1$, and so there is a higher probability that an edge is present between node $i$ and node $j$. If $p$ is small, the probability is lower.

\subsection{MAXIMUM-LIKELIHOOD ESTIMATION}
![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-495.jpg?height=372&width=1314&top_left_y=238&top_left_x=226)

(a) Graph representations of Erdős-Rényi graphs at different $p$.

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-495.jpg?height=337&width=1316&top_left_y=647&top_left_x=226)

(b) Adjacent matrices of the corresponding graphs.

Figure 8.9: A single-membership Erdös-Rényi graph is a graph structure in which the edge between node $i$ and node $j$ is defined as a Bernoulli random variable with parameter $p$. As $p$ increases, the graph has a higher probability of having more edges. The adjacent matrices shown in the bottom row are the mathematical representations of the graphs.

Suppose that we are given one snapshot of the network, i.e., one realization $\mathbf{x} \in R^{N \times N}$ of the adjacency matrix $\mathbf{X} \in \mathbb{R}^{N \times N}$. The problem of recovering the latent parameter $p$ can be formulated as an ML estimation.

Example 8.5. Write down the log-likelihood function of the single-membership ErdősRényi graph ML estimation problem.

Solution. Based on the definition of the graph model, we know that

$$
X_{i j} \sim \operatorname{Bernoulli}(p)
$$

Therefore, the probability mass function of $X_{i j}$ is

$$
\mathbb{P}\left[X_{i j}=1\right]=p \quad \text { and } \quad \mathbb{P}\left[X_{i j}=0\right]=1-p .
$$

This can be compactly expressed as

$$
f_{\mathbf{X}}(\mathbf{x} ; p)=\prod_{i=1}^{N} \prod_{j=1}^{N} p^{x_{i j}}(1-p)^{1-x_{i j}}
$$

Hence, the log-likelihood is

$$
\log \mathcal{L}(p \mid \mathbf{x})=\sum_{i=1}^{N} \sum_{j=1}^{N}\left\{x_{i j} \log p+\left(1-x_{i j}\right) \log (1-p)\right\} .
$$



\section{CHAPTER 8. ESTIMATION}

Now that we have the log-likelihood function, we can proceed to estimate the parameter $p$. The solution to this is the ML estimate.

Practice Exercise 8.3. Solve the ML estimation problem:

$$
\widehat{p}_{\mathrm{ML}}=\underset{p}{\operatorname{argmax}} \log \mathcal{L}(p \mid \mathbf{x})
$$

Solution. Using the log-likelihood we just derived, we have that

$$
\widehat{p}_{\mathrm{ML}}=\sum_{i=1}^{N} \sum_{j=1}^{N}\left\{x_{i j} \log p+\left(1-x_{i j}\right) \log (1-p)\right\}
$$

Taking the derivative and setting it to zero,

$$
\begin{aligned}
\frac{d}{d p} \log \mathcal{L}(p \mid \mathbf{x}) & =\frac{d}{d p}\left\{\sum_{i=1}^{N} \sum_{j=1}^{N}\left\{x_{i j} \log p+\left(1-x_{i j}\right) \log (1-p)\right\}\right\} \\
& =\sum_{i=1}^{N} \sum_{j=1}^{N}\left\{\frac{x_{i j}}{p}-\frac{1-x_{i j}}{1-p}\right\}=0
\end{aligned}
$$

Let $S=\sum_{i=1}^{N} \sum_{j=1}^{N} x_{i j}$. The equation above then becomes

$$
\frac{S}{p}-\frac{N^{2}-S}{1-p}=0
$$

Rearranging the terms yields $(1-p) S=p\left(N^{2}-S\right)$, which gives us

$$
\widehat{p}_{\mathrm{ML}}=\frac{S}{N^{2}}=\frac{1}{N^{2}} \sum_{i=1}^{N} \sum_{j=1}^{N} x_{i j} .
$$

On computers, visualizing the graphs and computing the ML estimates are reasonably straightforward. In MATLAB, you can call the command graph to build a graph from the adjacency matrix A. This will allow you to plot the graph. The computation, however, is done directly by the adjacency matrix. In the code below, you can see that we call rand to generate the Bernoulli random variables. The command triu extracts the upper triangular matrix from the matrix A. This ensures that we do not pick the diagonals. The symmetrization of $\mathrm{A}+\mathrm{A}^{\prime}$ ensures that the graph is indirectional, meaning that $i$ to $j$ is the same as $j$ to $i$.

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-496.jpg?height=317&width=698&top_left_y=1934&top_left_x=149)



\subsection{MAXIMUM-LIKELIHOOD ESTIMATION}

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-497.jpg?height=117&width=1392&top_left_y=244&top_left_x=210)

In Python, the computation is done similarly with the help of the networkx library. The number of edges $m$ is defined as $m=p \frac{n^{2}}{2}$. This is because for a graph with $n$ nodes, there are at most $\frac{n^{2}}{2}$ unique pairs of indirected edges. Multiplying this number by the probability $p$ will give us the number of edges $m$.

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-497.jpg?height=444&width=868&top_left_y=558&top_left_x=209)

As you can see in both the MATLAB and the Python code, the ML estimate $\widehat{p}_{\mathrm{ML}}$ is determined by taking the sample average. Thus the ML estimate, according to our calculation, is $\widehat{p}_{\mathrm{ML}}=\frac{1}{N^{2}} \sum_{i=1}^{N} \sum_{j=1}^{N} x_{i j}$.

\subsubsection{Application 2: Reconstructing images}

Being able to see in the dark is the holy grail of imaging. Many advanced sensing technologies have been developed over the past decade. In this example, we consider a single-photon image sensor. This is a counting device that counts the number of photons arriving at the sensor. Physicists have shown that a Poisson process can model the arrival of the photons. For simplicity we assume a homogeneous pattern of $N$ pixels. The underlying intensity of the homogeneous pattern is a constant $\lambda$.

Suppose that we have a sensor with $N$ pixels $x^{(1)}, \ldots, x^{(n)}$. According to the Poisson statistics, the probability of observing a pixel value is determined by the Poisson probability:

$$
x^{(n)} \sim \operatorname{Poisson}(\lambda), \quad n=1, \ldots, N
$$

or more explicitly,

$$
\mathbb{P}\left[x^{(n)}=x^{(n)}\right]=\frac{\lambda^{x^{(n)}}}{x^{(n)} !} e^{-\lambda}
$$

where $x^{(n)}$ is the $n$th observed pixel value, and is an integer.

A single-photon image sensor is slightly more complicated in the sense that it does not report $x^{(n)}$ but instead reports a truncated version of $x^{(n)}$. Depending on the number of incoming photons, the sensor reports

$$
Y_{n}= \begin{cases}1, & x^{(n)} \geq 1 \\ 0, & x^{(n)}=0 .\end{cases}
$$

We call this type of sensors a one-bit single-photon image sensor (see Figure 8.10. Our question is: If we are given the measurements $x^{(1)}, \ldots, x^{(n)}$, can we estimate the underlying parameter $\lambda$ ?

\section{CHAPTER 8. ESTIMATION}

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-498.jpg?height=435&width=1404&top_left_y=232&top_left_x=138)

Figure 8.10: A one-bit single-photon image sensor captures an image with binary bits: It reports a "1" when the number of photons exceeds certain threshold, and "0" otherwise. The recovery problem here is to estimate the underlying image from the measurements.

Example 8.6. Derive the log-likelihood function of the estimation problem for the single-photon image sensors.

Solution. Since $Y_{n}$ is a binary random variable, its probability is completely specified by the two states it takes:

$$
\begin{aligned}
& \mathbb{P}\left[Y_{n}=0\right]=\mathbb{P}\left[x^{(n)}=0\right]=e^{-\lambda} \\
& \mathbb{P}\left[Y_{n}=1\right]=\mathbb{P}\left[x^{(n)} \neq 0\right]=1-e^{-\lambda} .
\end{aligned}
$$

Thus, $Y_{n}$ is a Bernoulli random variable with probability $1-e^{-\lambda}$ of getting a value of 1 , and probability $e^{-\lambda}$ of getting a value of 0 . By defining $y_{n}$ as a binary number taking values of either 0 or 1 , it follows that the log-likelihood is

$$
\begin{aligned}
\log \mathcal{L}(\lambda \mid \boldsymbol{y}) & =\log \left\{\prod_{n=1}^{N}\left(1-e^{-\lambda}\right)^{y_{n}}\left(e^{-\lambda}\right)^{1-y_{n}}\right\} \\
& =\sum_{n=1}^{N}\left\{y_{n} \log \left(1-e^{-\lambda}\right)-\lambda\left(1-y_{n}\right)\right\} .
\end{aligned}
$$

Practice Exercise 8.4. Solve the ML estimation problem

$$
\widehat{\lambda}_{\mathrm{ML}}=\underset{\lambda}{\operatorname{argmax}} \log \mathcal{L}(\lambda \mid \boldsymbol{y})
$$

Solution. First, we define $S=\sum_{n=1}^{N} y_{n}$. This simplifies the log-likelihood function to

$$
\begin{aligned}
\log \mathcal{L}(\lambda \mid \boldsymbol{y}) & =\sum_{n=1}^{N}\left\{y_{n} \log \left(1-e^{-\lambda}\right)-\lambda\left(1-y_{n}\right)\right\} \\
& =S \log \left(1-e^{-\lambda}\right)-\lambda(N-S) .
\end{aligned}
$$



\subsection{MAXIMUM-LIKELIHOOD ESTIMATION}

The ML estimation is

$$
\widehat{\lambda}_{\mathrm{ML}}=\underset{\lambda}{\operatorname{argmax}} S \log \left(1-e^{-\lambda}\right)-\lambda(N-S)
$$

Taking the derivative w.r.t. $\lambda$ yields

$$
\frac{d}{d \lambda}\left\{S \log \left(1-e^{-\lambda}\right)-\lambda(N-S)\right\}=\frac{S}{1-e^{-\lambda}} e^{-\lambda}-(N-S) .
$$

Moving around the terms, it follows that

$$
\frac{S}{1-e^{-\lambda}} e^{-\lambda}-(N-S)=0 \quad \Longrightarrow \quad \lambda=-\log \left(1-\frac{S}{N}\right)
$$

Therefore, the ML estimate is

$$
\widehat{\lambda}_{\mathrm{ML}}=-\log \left(1-\frac{1}{N} \sum_{n=1}^{N} y_{n}\right)
$$

For real images, you can extrapolate the idea from $y_{n}$ to $y_{i, j, t}$, which denotes the $(i, j)$ th pixel located at time $t$. Defining $\boldsymbol{y}_{t} \in \mathbb{R}^{N \times N}$ as the $t$ th frame of the observed data, we can use $T$ frames to recover one image $\widehat{\boldsymbol{\lambda}}_{\mathrm{ML}} \in \mathbb{R}^{N \times N}$. It follows from the above derivation that the ML estimate is

$$
\widehat{\boldsymbol{\lambda}}_{\mathrm{ML}}=-\log \left(1-\frac{1}{T} \sum_{t=1}^{T} \boldsymbol{y}_{t}\right)
$$

Figure 8.11 shows a pair of input-output images of a $256 \times 256$ image.

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-499.jpg?height=500&width=493&top_left_y=1388&top_left_x=366)

(a) Observed data (1-frame)

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-499.jpg?height=498&width=496&top_left_y=1389&top_left_x=914)

(b) ML estimate (using 100 frames)

Figure 8.11: ML estimation for a single-photon image sensor problem. The observed data consists of 100 frames of binary measurements $\boldsymbol{y}_{1}, \ldots, \boldsymbol{y}_{T}$, where $T=100$. The ML estimate is constructed by $\boldsymbol{\lambda}=-\log \left(1-\frac{1}{T} \sum_{t=1}^{T} \boldsymbol{y}_{t}\right)$

On a computer the ML estimation can be done in a few lines of MATLAB code. The code in Python requires more work, as it needs to read images using the openCV library.

\subsection{MAXIMUM-LIKELIHOOD ESTIMATION}

To solve the ML estimation problem, we maximize the log-likelihood:

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\mathrm{ML}} \stackrel{\text { def }}{=} \underset{\boldsymbol{\theta}}{\operatorname{argmax}} & \mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x}) \\
& =\underset{\mu, \sigma^{2}}{\operatorname{argmax}}\left\{-\frac{N}{2} \log \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(x^{(n)}-\mu\right)^{2}\right\} .
\end{aligned}
$$

Since we have two parameters, we need to take the derivatives for both.

$$
\begin{aligned}
\frac{d}{d \mu}\left\{-\frac{N}{2} \log \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(x^{(n)}-\mu\right)^{2}\right\} & =0, \\
\frac{d}{d \sigma^{2}}\left\{-\frac{N}{2} \log \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(x^{(n)}-\mu\right)^{2}\right\} & =0 .
\end{aligned}
$$

(Note that the derivative of the second equation is taken w.r.t. to $\sigma^{2}$ and not $\sigma$.) This pair of equations gives us

$$
\frac{1}{\sigma^{2}} \sum_{n=1}^{N}\left(x^{(n)}-\mu\right)=0, \text { and }-\frac{N}{2} \cdot \frac{1}{2 \pi \sigma^{2}} \cdot(2 \pi)+\frac{1}{2 \sigma^{4}} \sum_{n=1}^{N}\left(x^{(n)}-\mu\right)^{2}=0
$$

Rearranging the equations, we find that

$$
\widehat{\mu}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)} \quad \text { and } \quad \widehat{\sigma}_{\mathrm{ML}}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(x^{(n)}-\widehat{\mu}_{\mathrm{ML}}\right)^{2} .
$$

Practice Exercise 8.6. (Poisson) Given a set of i.i.d. Poisson random variables $x^{(1)}, \ldots, x^{(n)}$ with an unknown parameter $\lambda$, find the ML estimate of $\lambda$.

Solution. For a Poisson random variable, the likelihood function is

$$
\mathcal{L}(\lambda \mid \mathbf{x})=\prod_{n=1}^{N}\left\{\frac{\lambda^{x^{(n)}}}{x^{(n)} !} e^{-\lambda}\right\}
$$

To solve the ML estimation problem, we note that

$$
\begin{aligned}
\widehat{\lambda}_{\mathrm{ML}}=\underset{\lambda}{\operatorname{argmax}} \mathcal{L}(\lambda \mid \mathbf{x}) & =\underset{\lambda}{\operatorname{argmax}} \log \left\{\prod_{n=1}^{N} \frac{\lambda^{x^{(n)}}}{x^{(n)} !} e^{-\lambda}\right\} \\
& =\underset{\lambda}{\operatorname{argmax}} \log \left\{\frac{\lambda \sum_{n} x^{(n)}}{\prod_{n} x^{(n)} !} e^{-N \lambda}\right\} .
\end{aligned}
$$

Since $\prod_{n} x^{(n)}$ ! is independent of $\lambda$, its presence or absence will not affect the optimization

\section{CHAPTER 8. ESTIMATION}

problem. Consequently we can drop the term. It follows that

$$
\begin{aligned}
\widehat{\lambda}_{\mathrm{ML}} & =\underset{\lambda}{\operatorname{argmax}} \log \left\{\lambda^{\sum_{n} x^{(n)}} e^{-N \lambda}\right\} \\
& =\underset{\lambda}{\operatorname{argmax}}\left(\sum_{n} x^{(n)}\right) \log \lambda-N \lambda .
\end{aligned}
$$

Taking the derivative and setting it to zero yields

$$
\frac{d}{d \lambda}\left\{\left(\sum_{n} x^{(n)}\right) \log \lambda-N \lambda\right\}=\frac{\sum_{n} x^{(n)}}{\lambda}-N=0
$$

Rearranging the terms yields

$$
\widehat{\lambda}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}
$$

The idea of ML estimation can also be extended to vector observations.

Example 8.7. (High-dimensional Gaussian) Suppose that we are given a set of i.i.d. $d$-dimensional Gaussian random vectors $\mathbf{X}_{1}, \ldots, \mathbf{X}_{N}$ such that

$$
\mathbf{X}_{n} \sim \operatorname{Gaussian}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

We assume that $\boldsymbol{\Sigma}$ is fixed and known, but $\boldsymbol{\mu}$ is unknown. Find the ML estimate of $\boldsymbol{\mu}$.

Solution. The likelihood function is

$$
\begin{aligned}
\mathcal{L}\left(\boldsymbol{\mu} \mid\left\{\mathbf{x}_{n}\right\}_{n=1}^{N}\right) & =\prod_{n=1}^{N} f_{\mathbf{X}_{n}}\left(\mathbf{x}_{n} ; \boldsymbol{\mu}\right) \\
& =\prod_{n=1}^{N}\left\{\frac{1}{\sqrt{(2 \pi)^{d}|\boldsymbol{\Sigma}|}} \exp \left\{-\frac{1}{2}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)\right\}\right\} \\
& =\left(\frac{1}{\sqrt{(2 \pi)^{d}|\boldsymbol{\Sigma}|}}\right)^{N} \exp \left\{-\frac{1}{2} \sum_{n=1}^{N}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)\right\} .
\end{aligned}
$$

Thus the log-likelihood function is

$$
\log \mathcal{L}\left(\boldsymbol{\mu} \mid\left\{\mathbf{x}_{n}\right\}_{n=1}^{N}\right)=\frac{N}{2} \log |\boldsymbol{\Sigma}|+\frac{N}{2} \log (2 \pi)^{d}+\sum_{n=1}^{N}\left\{\frac{1}{2}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)\right\} .
$$

The ML estimate is found by maximizing this log-likelihood function:

$$
\widehat{\boldsymbol{\mu}}_{\mathrm{ML}}=\underset{\boldsymbol{\mu}}{\operatorname{argmax}} \log \mathcal{L}\left(\boldsymbol{\mu} \mid\left\{\mathbf{x}_{n}\right\}_{n=1}^{N}\right) .
$$



\subsection{MAXIMUM-LIKELIHOOD ESTIMATION}

Taking the gradient of the function and setting it to zero, we have that

$$
\frac{d}{d \boldsymbol{\mu}}\left\{\frac{N}{2} \log |\boldsymbol{\Sigma}|+\frac{N}{2} \log (2 \pi)^{d}+\sum_{n=1}^{N}\left\{\frac{1}{2}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)\right\}\right\}=0
$$

The derivatives of the first two terms are zero because they do not depend on $\boldsymbol{\mu}$ ). Thus we have that:

$$
\sum_{n=1}^{N}\left\{\boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)\right\}=0
$$

Rearranging the terms yields the ML estimate $\widehat{\boldsymbol{\mu}}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_{n}$.

Example 8.8. (High-dimensional Gaussian) Assume the same problem setting as in Example 8.7, except that this time we assume that both the mean vector $\boldsymbol{\mu}$ and the covariance matrix $\boldsymbol{\Sigma}$ are unknown. Find the ML estimate for $\boldsymbol{\theta}=(\boldsymbol{\mu}, \boldsymbol{\Sigma})$.

Solution. The log-likelihood follows from Example 8.7:

$$
\log \mathcal{L}\left(\boldsymbol{\mu} \mid\left\{\mathbf{x}_{n}\right\}_{n=1}^{N}\right)=\frac{N}{2} \log |\boldsymbol{\Sigma}|+\frac{N}{2} \log (2 \pi)^{d}+\sum_{n=1}^{N}\left\{\frac{1}{2}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)\right\}
$$

Finding the ML estimate requires taking the derivative with respect to both $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ :

$$
\begin{aligned}
& \frac{d}{d \boldsymbol{\mu}}\left\{\frac{N}{2} \log |\boldsymbol{\Sigma}|+\frac{N}{2} \log (2 \pi)^{d}+\sum_{n=1}^{N}\left\{\frac{1}{2}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)\right\}\right\}=0 \\
& \frac{d}{d \boldsymbol{\Sigma}}\left\{\frac{N}{2} \log |\boldsymbol{\Sigma}|+\frac{N}{2} \log (2 \pi)^{d}+\sum_{n=1}^{N}\left\{\frac{1}{2}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)\right\}\right\}=0 .
\end{aligned}
$$

After some tedious algebraic steps (see Duda et al., Pattern Classification, Problem 3.14), we have that

$$
\begin{aligned}
& \widehat{\boldsymbol{\mu}}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_{n}, \\
& \widehat{\boldsymbol{\Sigma}}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N}\left(\mathbf{x}_{n}-\widehat{\boldsymbol{\mu}}_{\mathrm{ML}}\right)\left(\mathbf{x}_{n}-\widehat{\boldsymbol{\mu}}_{\mathrm{ML}}\right)^{T} .
\end{aligned}
$$

\subsubsection{Regression versus $\mathrm{ML}$ estimation}

ML estimation is closely related to regression. To understand the connection, we consider a linear model that we studied in Chapter 7. This model describes the relationship between

\section{CHAPTER 8. ESTIMATION}

the inputs $\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}$ and the observed outputs $y_{1}, \ldots, y_{N}$, via the equation

$$
y_{n}=\sum_{p=0}^{d-1} \theta_{p} \phi_{p}\left(\mathbf{x}_{n}\right)+e_{n}, \quad n=1, \ldots, N .
$$

In this expression, $\phi_{p}(\cdot)$ is a transformation that extracts the "features" of the input vector $\mathbf{x}$ to produce a scalar. The coefficient $\theta_{p}$ defines the relative weight of the feature $\phi_{p}\left(\mathbf{x}_{n}\right)$ in constructing the observed variable $y_{n}$. The error $e_{n}$ defines the modeling error between the observation $y_{n}$ and the prediction $\sum_{p=0}^{d-1} \theta_{p} \phi_{p}\left(\mathbf{x}_{n}\right)$. We call this equation a linear model.

Expressed in matrix form, the linear model is

$$
\underbrace{\left[\begin{array}{c}
y_{1} \\
y_{2} \\
\vdots \\
y_{N}
\end{array}\right]}_{=\boldsymbol{y}}=\underbrace{\left[\begin{array}{cccc}
\phi_{0}\left(\mathbf{x}_{1}\right) & \phi_{1}\left(\mathbf{x}_{1}\right) & \cdots & \phi_{d-1}\left(\mathbf{x}_{1}\right) \\
\phi_{0}\left(\mathbf{x}_{2}\right) & \phi_{1}\left(\mathbf{x}_{2}\right) & \cdots & \phi_{d-1}\left(\mathbf{x}_{2}\right) \\
\vdots & \ldots & \vdots & \vdots \\
\phi_{0}\left(\mathbf{x}_{N}\right) & \phi_{1}\left(\mathbf{x}_{N}\right) & \cdots & \phi_{d-1}\left(\mathbf{x}_{N}\right)
\end{array}\right]}_{=\mathbf{X}} \underbrace{\left[\begin{array}{c}
\theta_{0} \\
\theta_{1} \\
\vdots \\
\theta_{d-1}
\end{array}\right]}_{=\boldsymbol{\theta}}+\underbrace{\left[\begin{array}{c}
e_{1} \\
e_{2} \\
\vdots \\
e_{N}
\end{array}\right]}_{=\boldsymbol{e}},
$$

or more compactly as $\boldsymbol{y}=\mathbf{X} \boldsymbol{\theta}+\boldsymbol{e}$. Rearranging the terms, it is easy to show that

$$
\begin{aligned}
\sum_{n=1}^{N} e_{n}^{2} & =\sum_{n=1}^{N}\left(y_{n}-\sum_{p=0}^{d-1} \theta_{p} \phi_{p}\left(\mathbf{x}_{n}\right)\right)^{2} \\
& =\sum_{n=1}^{N}\left(y_{n}-[\mathbf{X} \boldsymbol{\theta}]_{n}\right)^{2}=\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2} .
\end{aligned}
$$

Now we make an assumption: that each noise $e_{n}$ is an i.i.d. copy of a Gaussian random variable with zero mean and variance $\sigma^{2}$. In other words, the error vector $\boldsymbol{e}$ is distributed according to $e \sim \operatorname{Gaussian}\left(\mathbf{0}, \sigma^{2} \boldsymbol{I}\right)$. This assumption is not always true because there are many situations in which the error is not Gaussian. However, this assumption is necessary for us to make the connection between ML estimation and regression.

With this assumption, we ask, given the observations $y_{1}, \ldots, y_{N}$, what would be the ML estimate of the unknown parameter $\boldsymbol{\theta}$ ? We answer this question in two steps.

Example 8.9. Find the likelihood function of $\boldsymbol{\theta}$, given $\boldsymbol{y}=\left[y_{1}, \ldots, y_{N}\right]^{T}$.

Solution. The PDF of $\boldsymbol{y}$ is given by a Gaussian:

$$
\begin{aligned}
f_{\boldsymbol{Y}}(\boldsymbol{y} ; \boldsymbol{\theta}) & =\prod_{n=1}^{N}\left\{\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{\left(y_{n}-[\mathbf{X} \boldsymbol{\theta}]_{n}\right)^{2}}{2 \sigma^{2}}\right\}\right\} \\
& =\frac{1}{\sqrt{\left(2 \pi \sigma^{2}\right)^{N}}} \exp \left\{-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(y_{n}-[\mathbf{X} \boldsymbol{\theta}]_{n}\right)^{2}\right\} \\
& =\frac{1}{\sqrt{\left(2 \pi \sigma^{2}\right)^{N}}} \exp \left\{-\frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}\right\}
\end{aligned}
$$



\subsection{MAXIMUM-LIKELIHOOD ESTIMATION}

Therefore, the log-likelihood function is

$$
\begin{aligned}
\log \mathcal{L}(\boldsymbol{\theta} \mid \boldsymbol{y}) & =\log \left\{\frac{1}{\sqrt{\left(2 \pi \sigma^{2}\right)^{N}}} \exp \left\{-\frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}\right\}\right\} \\
& =-\frac{N}{2} \log \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2} .
\end{aligned}
$$

The next step is to solve the ML estimation by maximizing the log-likelihood.

Example 8.10. Solve the ML estimation problem stated in Example 8.9. Assume that $\mathbf{X}^{T} \mathbf{X}$ is invertible.

Solution.

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\mathrm{ML}} & =\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \log \mathcal{L}(\boldsymbol{\theta} \mid \boldsymbol{y}) \\
& =\underset{\boldsymbol{\theta}}{\operatorname{argmax}}\left\{-\frac{N}{2} \log \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}\right\} .
\end{aligned}
$$

Taking the derivative w.r.t. $\boldsymbol{\theta}$ yields

$$
\frac{d}{d \boldsymbol{\theta}}\left\{-\frac{N}{2} \log \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}\right\}=0 .
$$

Since $\frac{d}{d \boldsymbol{\theta}} \boldsymbol{\theta}^{T} \boldsymbol{A} \boldsymbol{\theta}=\boldsymbol{A}+\boldsymbol{A}^{T}$, it follows from the chain rule that

$$
\begin{aligned}
\frac{d}{d \boldsymbol{\theta}}\left\{-\frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}\right\} & =\frac{d}{d \boldsymbol{\theta}}\left\{-\frac{1}{2 \sigma^{2}}(\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta})^{T}(\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta})\right\} \\
& =\frac{1}{\sigma^{2}} \mathbf{X}^{T}(\mathbf{X} \boldsymbol{\theta}-\boldsymbol{y}) .
\end{aligned}
$$

Substituting this result into the equation,

$$
\frac{1}{\sigma^{2}} \mathbf{X}^{T}(\mathbf{X} \boldsymbol{\theta}-\boldsymbol{y})=0
$$

Rearranging terms we obtain $\mathbf{X}^{T} \mathbf{X} \boldsymbol{\theta}=\mathbf{X}^{T} \boldsymbol{y}$, of which the solution is

$$
\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}=\left(\mathbf{X}^{T} \mathbf{X}\right)^{-1} \mathbf{X}^{T} \boldsymbol{y}
$$

Since the ML estimate in Equation 8.21) is the same as the regression solution (see Chapter 7), we conclude that the regression problem of a linear model is equivalent to solving an ML estimation problem.

The main difference between a linear regression problem and an ML estimation problem is the underlying statistical model, as illustrated in Figure 8.12. In linear regression, you do not care about the statistics of the noise term $e_{n}$. We choose $(\cdot)^{2}$ as the error because it is differentiable and convenient. In ML estimation, we choose $(\cdot)^{2}$ as the error because the noise is Gaussian. If the noise is not Gaussian, e.g., the noise follows a Laplace distribution, we need to choose $|\cdot|$ as the error. Therefore, you can always get a result by solving the linear regression. However, this result will only become meaningful if you provide additional

\section{CHAPTER 8. ESTIMATION}

\section{Regression}

Optimization:

$\widehat{\boldsymbol{\theta}}=\underset{\boldsymbol{\theta}}{\operatorname{argmin}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}$

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-506.jpg?height=76&width=103&top_left_y=470&top_left_x=195)

Solution:

$$
\widehat{\boldsymbol{\theta}}=\left(\mathbf{X}^{T} \mathbf{X}\right)^{-1} \mathbf{X}^{T} \boldsymbol{y}
$$

Assumption: None

\section{Maximum-Likelihood}

Optimization:
$\widehat{\boldsymbol{\theta}}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}}\left(\frac{1}{\sqrt{\left(2 \pi \sigma^{2}\right)^{d}}}\right)^{N} \exp \left\{-\frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}\right\}$

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-506.jpg?height=78&width=101&top_left_y=471&top_left_x=760)

Solution:

$\widehat{\boldsymbol{\theta}}=\left(\mathbf{X}^{T} \mathbf{X}\right)^{-1} \mathbf{X}^{T} \boldsymbol{y}$

Assumption: $\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta} \sim$ Gaussian $\left(0, \sigma^{2} \boldsymbol{I}\right)$

Figure 8.12: $\mathrm{ML}$ estimation is equivalent to a linear regression when the underlying statistical model for ML estimation is a Gaussian. Specifically, if the error term $\boldsymbol{e}=\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}$ is an independent Gaussian vector with zero mean and covariance matrix $\sigma^{2} \boldsymbol{I}$, then the resulting ML estimation is the same as linear regression. If the underlying statistical model is not Gaussian, then solving the regression is equivalent to applying a Gaussian ML estimation to a non-Gaussian problem. This will still give us a result, but that result will not maximize the likelihood, and thus it will not have any statistical guarantee.

information about the problem. For example, if you know that the noise is Gaussian, then the regression solution is also the ML solution. This is a statistical guarantee.

In practice, of course, we do not know whether the noise is Gaussian or not. At this point we have two courses of action: (i) Use your prior knowledge/domain expertise to determine whether a Gaussian assumption makes sense, or (ii) select an alternative model and see if the alternative model fits the data better. In practice, we should also question whether maximizing the likelihood is what we want. We may have some knowledge and therefore prefer the parameter $\boldsymbol{\theta}$, e.g., we want a sparse solution so that $\boldsymbol{\theta}$ only contains a few non-zeros. In that case, maximizing the likelihood without any constraint may not be the solution we want.

\section{$\mathrm{ML}$ estimation versus regression}

- ML estimation requires a statistical assumption, whereas regression does not.

- Suppose that you use a linear model $y_{n}=\sum_{p=0}^{d-1} \theta_{p} \phi_{p}\left(\mathbf{x}_{n}\right)+e_{n}$ where $e_{n} \sim$ Gaussian $\left(0, \sigma^{2}\right)$, for $n=1, \ldots, N$.

- Then the likelihood function in the ML estimation is

$$
\mathcal{L}(\boldsymbol{\theta} \mid \boldsymbol{y})=\frac{1}{\sqrt{\left(2 \pi \sigma^{2}\right)^{N}}} \exp \left\{-\frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}\right\},
$$

- The ML estimate $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}$ is $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}=\left(\mathbf{X}^{T} \mathbf{X}\right)^{-1} \mathbf{X}^{T} \boldsymbol{y}$, which is exactly the same as the regression solution. If the above statistical assumptions do not hold, then the regression solution will not maximize the likelihood.

\subsection{PROPERTIES OF ML ESTIMATES}

\subsection{Properties of ML Estimates}

ML estimation is a very special type of estimation. Not all estimations are ML. If an estimate is ML, are there any theoretical properties we can analyze? For example, will ML estimates guarantee the recovery of the true parameter? If so, when will this happen? In this section we investigate these theoretical questions so that you will acquire a better understanding of the statistical nature of ML estimates 2

\subsubsection{Estimators}

We know that an ML estimate is defined as

$$
\widehat{\theta}(\mathbf{x})=\underset{\theta}{\operatorname{argmax}} \mathcal{L}(\theta \mid \mathbf{x})
$$

We write $\widehat{\theta}(\mathbf{x})$ to emphasize that $\widehat{\theta}$ is a function of $\mathbf{x}$. The dependency of $\widehat{\theta}(\mathbf{x})$ on $\mathbf{x}$ should not be a surprise. For example, if the ML estimate is the sample average, we have that

$$
\widehat{\theta}\left(x^{(1)}, \ldots, x^{(n)}\right)=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}
$$

where $\mathbf{x}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$.

However, in this setting we should always remember that $x^{(1)}, \ldots, x^{(n)}$ are realizations of the i.i.d. random variables $x^{(1)}, \ldots, x^{(n)}$. Therefore, if we want to analzye the randomness of the variables, it is more reasonable to write $\widehat{\theta}$ as a random variable $\widehat{\Theta}_{\mathrm{ML}}$. For example, in the case of sample average, we have that

$$
\widehat{\Theta}_{\mathrm{ML}}\left(x^{(1)}, \ldots, x^{(n)}\right)=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}
$$

We call $\widehat{\Theta}_{\mathrm{ML}}$ the ML estimator of the true parameter $\theta$.

Estimate versus estimator

- An estimate is a number, e.g., $\widehat{\theta}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}$. It is the random realization of a random variable.

- An estimator is a random variable, e.g., $\widehat{\Theta}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}$. It takes a set of random variables and generates another random variable.

${ }^{2}$ For notational simplicity, in this section we will focus on a scalar parameter $\theta$ instead of a vector parameter $\boldsymbol{\theta}$.

\section{CHAPTER 8. ESTIMATION}

The ML estimators are one type of estimator, namely those that maximize the likelihood functions. If we do not want to maximize the likelihood we can still define an estimator. An estimator is any function that takes the data points $x^{(1)}, \ldots, x^{(n)}$ and maps them to a number (or a vector of numbers). That is, an estimator is

$$
\widehat{\Theta}\left(x^{(1)}, \ldots, x^{(n)}\right)
$$

We call $\widehat{\Theta}$ the estimator of the true parameter $\theta$.

Example 8.11. Let $x^{(1)}, \ldots, x^{(n)}$ be Gaussian i.i.d. random variables with unknown mean $\theta$ and known variance $\sigma^{2}$. Construct two possible estimators.

Solution. We define two estimators:

$$
\begin{aligned}
& \widehat{\Theta}_{1}\left(x^{(1)}, \ldots, x^{(n)}\right)=\frac{1}{N} \sum_{n=1}^{N} x^{(n)} \\
& \widehat{\Theta}_{2}\left(x^{(1)}, \ldots, x^{(n)}\right)=x^{(1)}
\end{aligned}
$$

In the first case, the estimator takes all the samples and constructs the sample average. The second estimator takes all the samples and returns on the first element. Both are legitimate estimators. However, $\widehat{\Theta}_{1}$ is the ML estimator, whereas $\widehat{\Theta}_{2}$ is not.

\subsubsection{Unbiased estimators}

While you can define estimators in any way you like, certain estimators are good and others are bad. By "good" we mean that the estimator can provide you with the information about the true parameter $\theta$; otherwise, why would you even construct such an estimator? However, the difficulty here is that $\widehat{\Theta}$ is a random variable because it is constructed from $x^{(1)}, \ldots, x^{(n)}$. Therefore, we need to define different metrics to quantify the usefulness of the estimators.

Definition 8.5. An estimator $\widehat{\Theta}$ is unbiased if

$$
\mathbb{E}[\widehat{\Theta}]=\theta
$$

Unbiasedness means that the average of the random variable $\widehat{\Theta}$ matches the true parameter $\theta$. In other words, while we allow $\widehat{\Theta}$ to fluctuate, we expect the average to match the true $\theta$. If this is not the case, using more measurements will not help us get closer to $\theta$.

Example 8.12. Let $x^{(1)}, \ldots, x^{(n)}$ be i.i.d. Gaussian random variables with a unknown mean $\theta$. It has been shown that the ML estimator is

$$
\widehat{\Theta}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)} .
$$

Is the ML estimator $\widehat{\Theta}_{\mathrm{ML}}$ unbiased?

\subsection{PROPERTIES OF ML ESTIMATES}

Solution: To check the unbiasedness, we look at the expectation:

$$
\mathbb{E}\left[\widehat{\Theta}_{\mathrm{ML}}\right]=\frac{1}{N} \sum_{n=1}^{N} \mathbb{E}\left[x^{(n)}\right]=\frac{1}{N} \sum_{n=1}^{N} \theta=\theta
$$

Thus, $\widehat{\Theta}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}$ is an unbiased estimator of $\theta$.

Example 8.13. Same as the example before, but this time we consider an estimator

$$
\widehat{\Theta}=x^{(1)}+x^{(2)}+5
$$

Is this estimator unbiased?

Solution: In this case,

$$
\mathbb{E}[\widehat{\Theta}]=\mathbb{E}\left[x^{(1)}+x^{(2)}+5\right]=\mathbb{E}\left[x^{(1)}\right]+\mathbb{E}\left[x^{(2)}\right]+5=2 \theta+5 \neq \theta
$$

Therefore, the estimator is biased.

Example 8.14. Let $x^{(1)}, \ldots, x^{(n)}$ be i.i.d. Gaussian random variables with unknown mean $\mu$ and unknown variance $\sigma^{2}$. We have shown that the ML estimators are

$$
\widehat{\mu}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)} \quad \text { and } \quad \widehat{\sigma}_{\mathrm{ML}}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(x^{(n)}-\widehat{\mu}_{\mathrm{ML}}\right)^{2} .
$$

It is easy to show that $\mathbb{E}\left[\widehat{\mu}_{\mathrm{ML}}\right]=\mu$. How about $\widehat{\sigma}_{\mathrm{ML}}^{2}$ ? Is it an unbiased estimator?

Solution: For simplicity we assume $\mu=0$ so that $\mathbb{E}\left[x^{(n)}^{2}\right]=\mathbb{E}\left[\left(x^{(n)}-0\right)^{2}\right]=\sigma^{2}$.

Note that

$$
\begin{aligned}
\mathbb{E}\left[\widehat{\sigma}_{\mathrm{ML}}^{2}\right] & =\frac{1}{N} \sum_{n=1}^{N}\left\{\mathbb{E}\left[x^{(n)}^{2}\right]-2 \mathbb{E}\left[\widehat{\mu}_{\mathrm{ML}} x^{(n)}\right]+\mathbb{E}\left[\widehat{\mu}_{\mathrm{ML}}^{2}\right]\right\} \\
& =\frac{1}{N} \sum_{n=1}^{N}\left\{\sigma^{2}-2 \mathbb{E}\left[\frac{1}{N} \sum_{j=1}^{N} X_{j} x^{(n)}\right]+\mathbb{E}\left[\left(\frac{1}{N} \sum_{n=1}^{N} x^{(n)}\right)^{2}\right]\right\} .
\end{aligned}
$$

By independence, we observe that $\mathbb{E}\left[X_{j} x^{(n)}\right]=\mathbb{E}\left[X_{j}\right] \mathbb{E}\left[x^{(n)}\right]=0$, for any $j \neq n$. Therefore,

$$
\begin{aligned}
\mathbb{E}\left[\frac{1}{N} \sum_{j=1}^{N} X_{j} x^{(n)}\right] & =\frac{1}{N} \mathbb{E}\left[x^{(1)} x^{(n)}+\cdots+x^{(n)} x^{(n)}\right] \\
& =\frac{1}{N}\left(0+\cdots+\sigma^{2}+\cdots+0\right)=\frac{\sigma^{2}}{N}
\end{aligned}
$$



\section{CHAPTER 8. ESTIMATION}

Similarly, we have that

$$
\begin{aligned}
\mathbb{E}\left[\left(\frac{1}{N} \sum_{n=1}^{N} x^{(n)}\right)^{2}\right] & =\frac{1}{N^{2}} \sum_{n=1}^{N}\left\{\mathbb{E}\left[x^{(n)}^{2}\right]+\sum_{j \neq n} \mathbb{E}\left[X_{j} x^{(n)}\right]\right\} \\
& =\frac{1}{N^{2}} \sum_{n=1}^{N}\left\{\sigma^{2}+0\right\}=\frac{\sigma^{2}}{N}
\end{aligned}
$$

Combining everything, we arrive at the result:

$$
\begin{aligned}
\mathbb{E}\left[\widehat{\sigma}_{\mathrm{ML}}^{2}\right] & =\frac{1}{N} \sum_{n=1}^{N}\left\{\sigma^{2}-2 \mathbb{E}\left[\frac{1}{N} \sum_{j=1}^{N} X_{j} x^{(n)}\right]+\mathbb{E}\left[\left(\frac{1}{N} \sum_{n=1}^{N} x^{(n)}\right)^{2}\right]\right\} \\
& =\frac{1}{N} \sum_{n=1}^{N}\left\{\sigma^{2}-\frac{2 \sigma^{2}}{N}+\frac{\sigma^{2}}{N}\right\} \\
& =\frac{N-1}{N} \sigma^{2}
\end{aligned}
$$

which is not equal to $\sigma^{2}$. Therefore, $\widehat{\sigma}_{\mathrm{ML}}^{2}$ is a biased estimator of $\sigma^{2}$.

In the previous example, it is possible to construct an unbiased estimator for the variance. To do so, we can use

$$
\widehat{\sigma}_{\text {unbias }}^{2}=\frac{1}{N-1} \sum_{n=1}^{N}\left(x^{(n)}-\widehat{\mu}_{\mathrm{ML}}\right)^{2},
$$

so that $\mathbb{E}\left[\widehat{\sigma}_{\text {unbias }}^{2}\right]=\sigma^{2}$. However, note that $\widehat{\sigma}_{\text {unbias }}^{2}$ does not maximize the likelihood, so while you can get unbiasedness, you cannot maximize the likelihood. If you want to maximize the likelihood, you cannot get unbiasedness.

What is an unbiased estimator?

- An estimator $\widehat{\Theta}$ is unbiased if $\mathbb{E}[\widehat{\Theta}]=\theta$.

- Unbiased means that the statistical average of $\widehat{\Theta}$ is the true parameter $\theta$.

- If $x^{(n)} \sim \operatorname{Gaussian}\left(\theta, \sigma^{2}\right)$, then $\widehat{\Theta}=(1 / N) \sum_{n=1}^{N} x^{(n)}$ is unbiased, but $\widehat{\Theta}=x^{(1)}$ is biased.

\subsubsection{Consistent estimators}

By definition, an estimator $\widehat{\Theta}\left(x^{(1)}, \ldots, x^{(n)}\right)$ is a function of $N$ random variables $x^{(1)}, \ldots, x^{(n)}$. Therefore, $\widehat{\Theta}\left(x^{(1)}, \ldots, x^{(n)}\right)$ changes as $N$ grows. In this subsection we analyze how $\widehat{\Theta}$ behaves when $N$ changes. For notational simplicity we use the following notation:

$$
\widehat{\Theta}_{N}=\widehat{\Theta}\left(x^{(1)}, \ldots, x^{(n)}\right)
$$

Thus, as $N$ increases, we use more random variables in defining $\widehat{\Theta}\left(x^{(1)}, \ldots, x^{(n)}\right)$.

\subsection{PROPERTIES OF ML ESTIMATES}

Definition 8.6. An estimator $\widehat{\Theta}_{N}$ is consistent if $\widehat{\Theta}_{N} \stackrel{p}{\longrightarrow} \theta$, i.e.,

$$
\lim _{N \rightarrow \infty} \mathbb{P}\left[\left|\widehat{\Theta}_{N}-\theta\right| \geq \epsilon\right]=0
$$

The definition here follows from our discussions of the law of large numbers in Chapter 6 . The specific type of convergence is known as the convergence in probability. It says that as $N$ grows, the estimator $\Theta$ will be close enough to $\theta$ so that the probability of getting a large deviation will diminish, as illustrated in Figure 8.13.

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-511.jpg?height=349&width=635&top_left_y=704&top_left_x=251)

(a) $N=1$

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-511.jpg?height=354&width=635&top_left_y=1102&top_left_x=251)

(c) $N=4$

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-511.jpg?height=351&width=629&top_left_y=706&top_left_x=916)

(b) $N=2$

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-511.jpg?height=352&width=629&top_left_y=1106&top_left_x=916)

(d) $N=8$

Figure 8.13: The four subfigures here illustrate the probability of error $\mathbb{P}\left[\left|\widehat{\Theta}_{N}-\theta\right| \geq \epsilon\right]$, which is represented by the areas shaded in blue. We assume that the estimator $\widehat{\Theta}_{N}$ is a Gaussian random variable following a distribution Gaussian $\left(0, \frac{\sigma^{2}}{N}\right)$, where we set $\sigma=1$. The threshold we use in this figure is $\epsilon=1$. As $N$ grows, we see that the probability of error diminishes. If the probability of error goes to zero, we say that the estimator is consistent.

The examples in Figure 8.13 are typical situations for an estimator based on the sample average. For example, if we assume that $x^{(1)}, \ldots, x^{(n)}$ are i.i.d. Gaussian copies of Gaussian $\left(0, \sigma^{2}\right)$, then the estimator

$$
\widehat{\Theta}\left(x^{(1)}, \ldots, x^{(n)}\right)=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}
$$

will follow a Gaussian distribution $\operatorname{Gaussian}\left(0, \frac{\sigma^{2}}{N}\right)$. (Please refer to Chapter 6 for the derivation.) Then, as $N$ grows, the PDF of $\widehat{\Theta}_{N}$ becomes narrower and narrower. For a fixed $\epsilon$, it follows that the probability of error will diminish to zero. In fact, we can prove that, for this

\section{CHAPTER 8. ESTIMATION}

example,

$$
\begin{aligned}
\mathbb{P}\left[\left|\widehat{\Theta}_{N}-\theta\right| \geq \epsilon\right] & =\mathbb{P}\left[\widehat{\Theta}_{N}-\theta \geq \epsilon\right]+\mathbb{P}\left[\widehat{\Theta}_{N}-\theta \leq-\epsilon\right] \\
& =\int_{\theta+\epsilon}^{\infty} \text { Gaussian }\left(z \mid \theta, \frac{\sigma^{2}}{N}\right) d z+\int_{-\infty}^{\theta-\epsilon} \text { Gaussian }\left(z \mid \theta, \frac{\sigma^{2}}{N}\right) d z \\
& =\int_{\theta+\epsilon}^{\infty} \frac{1}{\sqrt{2 \pi \sigma^{2} / N}} e^{-\frac{(z-\theta)^{2}}{2 \sigma^{2} / N}} d z+\int_{-\infty}^{\theta-\epsilon} \frac{1}{\sqrt{2 \pi \sigma^{2} / N}} e^{-\frac{(z-\theta)^{2}}{2 \sigma^{2} / N}} d z \\
& =\int_{\frac{\epsilon}{\sigma / \sqrt{N}}}^{\infty} \frac{1}{\sqrt{2 \pi}} e^{-\frac{z^{2}}{2}} d z+\int_{-\infty}^{-\frac{\epsilon}{\sigma / \sqrt{N}}} \frac{1}{\sqrt{2 \pi}} e^{-\frac{z^{2}}{2}} d z \\
& =1-\Phi\left(\frac{\epsilon}{\sigma / \sqrt{N}}\right)+\Phi\left(\frac{-\epsilon}{\sigma / \sqrt{N}}\right) \\
& =2 \Phi\left(\frac{-\epsilon}{\sigma / \sqrt{N}}\right) .
\end{aligned}
$$

Therefore, as $N \rightarrow \infty$, it holds that $\frac{-\epsilon}{\sigma / \sqrt{N}} \rightarrow-\infty$. Hence,

$$
\lim _{N \rightarrow \infty} \mathbb{P}\left[\left|\widehat{\Theta}_{N}-\theta\right| \geq \epsilon\right]=\lim _{N \rightarrow \infty} 2 \Phi\left(\frac{-\epsilon}{\sigma / \sqrt{N}}\right)=0 .
$$

This explains why in Figure $\mathbf{8 . 1 3}$ the probability of error diminishes to zero as $N$ grows. Therefore, we say that $\widehat{\Theta}_{N}$ is consistent.

In general, there are two ways to check whether an estimator is consistent:

- Prove convergence in probability. This is based on the definition of a consistent estimator. If we can prove that

$$
\lim _{N \rightarrow \infty} \mathbb{P}\left[\left|\widehat{\Theta}_{N}-\theta\right| \geq \epsilon\right]=0
$$

then we say that the estimator is consistent.

- Prove convergence in mean squared error:

$$
\lim _{N \rightarrow \infty} \mathbb{E}\left[\left(\widehat{\Theta}_{N}-\theta\right)^{2}\right]=0
$$

To see why convergence in the mean squared error is sufficient to guarantee consistency, we recall Chebyshev's inequality in Chapter 6 , which says that

$$
\mathbb{P}\left[\left|\widehat{\Theta}_{N}-\theta\right| \geq \epsilon\right] \leq \frac{\mathbb{E}\left[\left(\widehat{\Theta}_{N}-\theta\right)^{2}\right]}{\epsilon^{2}}
$$

Thus, if $\lim _{N \rightarrow \infty} \mathbb{E}\left[\left(\widehat{\Theta}_{N}-\theta\right)^{2}\right]=0$, convergence in probability will also hold. However, since mean square convergence is stronger than convergence in probability, being unable to show mean square convergence does not imply that an estimator is inconsistent.

Be careful not to confuse a consistent estimator and an unbiased estimator. The two are different concepts; one does not imply the other.

\subsection{PROPERTIES OF ML ESTIMATES}

\section{Consistent versus unbiased}

- Consistent $=$ If you have enough samples, then the estimator $\widehat{\Theta}$ will converge to the true parameter.

- Unbiasedness does not imply consistency. For example (Gaussian), if

$$
\widehat{\Theta}=x^{(1)}
$$

then $\mathbb{E}\left[x^{(1)}\right]=\mu$. But $\mathbb{P}[|\widehat{\Theta}-\mu|>\epsilon]$ does not converge to 0 as $N$ grows. So this estimator is inconsistent. (See Example 8.16 below.)

- Consistency does not imply unbiasedness. For example (Gaussian), if

$$
\widehat{\Theta}=\frac{1}{N} \sum_{n=1}^{N}\left(x^{(n)}-\mu\right)^{2}
$$

is a biased estimate for variance, but it is consistent. (See Example 8.17 below.)

Example 8.15. Let $x^{(1)}, \ldots, x^{(n)}$ be i.i.d. Gaussian random variables with an unknown mean $\mu$ and known variance $\sigma^{2}$. We know that the ML estimator for the mean is $\widehat{\mu}_{\mathrm{ML}}=(1 / N) \sum_{n=1}^{N} x^{(n)}$. Is $\widehat{\mu}_{\mathrm{ML}}$ consistent?

Solution. We have shown that the ML estimator is

$$
\widehat{\mu}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)} .
$$

Since $\mathbb{E}\left[\widehat{\mu}_{\mathrm{ML}}\right]=\mu$, and $\mathbb{E}\left[\left(\widehat{\mu}_{\mathrm{ML}}-\mu\right)^{2}\right]=\operatorname{Var}\left[\widehat{\mu}_{\mathrm{ML}}\right]=\frac{\sigma^{2}}{N}$, it follows that

$$
\mathbb{P}\left[\left|\widehat{\mu}_{\mathrm{ML}}-\mu\right| \geq \epsilon\right] \leq \frac{\mathbb{E}\left[\left(\widehat{\mu}_{\mathrm{ML}}-\mu\right)^{2}\right]}{\epsilon^{2}}=\frac{\sigma^{2}}{N \epsilon^{2}}
$$

Thus, when $N$ goes to infinity, the probability converges to zero, and hence the estimator is consistent.

Example 8.16. Let $x^{(1)}, \ldots, x^{(n)}$ be i.i.d. Gaussian random variables with an unknown mean $\mu$ and known variance $\sigma^{2}$. Define an estimator $\widehat{\mu}=x^{(1)}$. Show that the estimator is unbiased but inconsistent.

Solution. We know that $\mathbb{E}[\widehat{\mu}]=\mathbb{E}\left[x^{(1)}\right]=\mu$. So $\widehat{\mu}$ is an unbiased estimator. However, we can show that

$$
\mathbb{E}\left[(\widehat{\mu}-\mu)^{2}\right]=\mathbb{E}\left[\left(x^{(1)}-\mu\right)^{2}\right]=\sigma^{2}
$$

Since this variance $\mathbb{E}\left[(\widehat{\mu}-\mu)^{2}\right]$ does not shrink as $N$ increases, it follows that no matter

\section{CHAPTER 8. ESTIMATION}

how many samples we use we cannot make $\mathbb{E}\left[(\widehat{\mu}-\mu)^{2}\right]$ go to zero. To be more precise,

$$
\begin{aligned}
\mathbb{P}[|\widehat{\mu}-\mu| \geq \epsilon] & =\mathbb{P}\left[\left|x^{(1)}-\mu\right| \geq \epsilon\right] \\
& =\mathbb{P}\left[x^{(1)} \leq \mu-\epsilon\right]+\mathbb{P}\left[x^{(1)} \geq \mu+\epsilon\right] \\
& =\int_{-\infty}^{\mu-\epsilon} \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{(x-\mu)^{2}}{2 \sigma^{2}}} d x+\int_{\mu+\epsilon}^{\infty} \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{(x-\mu)^{2}}{2 \sigma^{2}}} d x \\
& =2 \Phi\left(\frac{-\epsilon}{\sigma}\right),
\end{aligned}
$$

which does not converge to zero as $N \rightarrow \infty$. So the estimator is inconsistent.

Example 8.17. Let $x^{(1)}, \ldots, x^{(n)}$ be i.i.d. Gaussian random variables with an unknown mean $\mu$ and an unknown variance $\sigma^{2}$. Is the ML estimate of the variance, i.e., $\widehat{\sigma}_{\mathrm{ML}}^{2}$, consistent?

Solution. We know that the ML estimator for the mean is

$$
\widehat{\mu}_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}
$$

and we have shown that it is an unbiased and consistent estimator of the mean. For the variance,

$$
\begin{aligned}
\widehat{\sigma}_{\mathrm{ML}}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(x^{(n)}-\widehat{\mu}_{\mathrm{ML}}\right)^{2} & =\frac{1}{N} \sum_{n=1}^{N}\left[x^{(n)}^{2}-2 \widehat{\mu}_{\mathrm{ML}} x^{(n)}+\widehat{\mu}_{\mathrm{ML}}^{2}\right] \\
& =\frac{1}{N} \sum_{n=1}^{N} x^{(n)}^{2}-2 \widehat{\mu}_{\mathrm{ML}} \cdot \frac{1}{N} \sum_{n=1}^{N} x^{(n)}+\widehat{\mu}_{\mathrm{ML}}^{2} \\
& =\frac{1}{N} \sum_{n=1}^{N} x^{(n)}^{2}-\widehat{\mu}_{\mathrm{ML}}^{2} .
\end{aligned}
$$

Note that $\frac{1}{N} \sum_{n=1}^{N} x^{(n)}^{2}$ is the sample average of the second moment, and so by the weak law of large numbers it should converge in probability to $\mathbb{E}\left[x^{(n)}^{2}\right]$. Similarly, $\widehat{\mu}_{\mathrm{ML}}$ will converge in probability to $\mu$. Therefore, we have

$$
\widehat{\sigma}_{\mathrm{ML}}^{2}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}^{2}-\widehat{\mu}_{\mathrm{ML}}^{2} \stackrel{p}{\longrightarrow}\left(\sigma^{2}+\mu^{2}\right)-\mu^{2}=\sigma^{2} .
$$

Thus, we have shown that the ML estimator of the variance is biased but consistent.

\subsection{PROPERTIES OF ML ESTIMATES}

The following discussions about the consistency of $\mathrm{ML}$ estimators can be skipped.

As we have said, there are many estimators. Some estimators are consistent and some are not. The ML estimators are special. It turns out that under certain regularity conditions the ML estimators of i.i.d. observations are consistent.

Without proving this result formally, we highlight a few steps to illustrate the idea. Suppose that we have a set of i.i.d. data points $\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}$ drawn from some distribution $f\left(\mathbf{x}, \mid \boldsymbol{\theta}_{\text {true }}\right)$. To formulate the ML estimation, we consider the log-likelihood function (divided by $N)$ :

$$
\frac{1}{N} \log \mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x})=\frac{1}{N} \sum_{n=1}^{N} \log f\left(\mathbf{x}_{n} ; \boldsymbol{\theta}\right)
$$

Here, the variable $\boldsymbol{\theta}$ is unknown. We need to find it by maximizing the log-likelihood.

By the weak law of large numbers, we can show that the log-likelihood based on the $N$ samples will converge in probability to

$$
\underbrace{\frac{1}{N} \sum_{n=1}^{N} \log f\left(\mathbf{x}_{n} ; \boldsymbol{\theta}\right)}_{g_{N}(\boldsymbol{\theta})} \stackrel{p}{\longrightarrow} \mathbb{E}[\log f(\mathbf{x} ; \boldsymbol{\theta})] .
$$

The expectation can be evaluated by integrating over the true distribution:

$$
\mathbb{E}[\log f(\mathbf{x} ; \boldsymbol{\theta})]=\underbrace{\int \log f(\mathbf{x} ; \boldsymbol{\theta}) \cdot f\left(\mathbf{x} ; \boldsymbol{\theta}_{\text {true }}\right) d \mathbf{x}}_{g(\boldsymbol{\theta})}
$$

where $f\left(\mathbf{x} ; \boldsymbol{\theta}_{\text {true }}\right)$ denotes the true distribution of the samples $\mathbf{x}_{n}$ 's. From these two results we define two functions:

$$
g_{N}(\boldsymbol{\theta}) \stackrel{\text { def }}{=} \frac{1}{N} \sum_{n=1}^{N} \log f\left(\mathbf{x}_{n} ; \boldsymbol{\theta}\right), \text { and } g(\boldsymbol{\theta}) \stackrel{\text { def }}{=} \int \log f(\mathbf{x} ; \boldsymbol{\theta}) \cdot f\left(\mathbf{x} ; \boldsymbol{\theta}_{\text {true }}\right) d \mathbf{x},
$$

and we know that $g_{N}(\boldsymbol{\theta}) \stackrel{p}{\longrightarrow} g(\boldsymbol{\theta})$.

We also know that $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}$ is the ML estimator, and so

$$
\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}} g_{N}(\boldsymbol{\theta})
$$

Let $\boldsymbol{\theta}^{*}$ be the maximizer of the limiting function, i.e.,

$$
\boldsymbol{\theta}^{*}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}} g(\boldsymbol{\theta})
$$

Because $g_{N}(\boldsymbol{\theta}) \stackrel{p}{\rightarrow} g(\boldsymbol{\theta})$, we can (loosely ${ }^{3}$ argue that $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}} \stackrel{p}{\rightarrow} \boldsymbol{\theta}^{*}$. If we can show that $\boldsymbol{\theta}^{*}=\boldsymbol{\theta}_{\text {true }}$, then we have shown that $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}} \stackrel{p}{\rightarrow} \boldsymbol{\theta}_{\text {true }}$, implying that $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}$ is consistent.

${ }^{3}$ To rigorously prove this statement we need some kind of regularity conditions on $g_{N}$ and $g$. A more formal proof can be found in H. Vincent Poor, An Introduction Signal Detection and Estimation, Springer, 1998, Section IV.D.

\section{CHAPTER 8. ESTIMATION}

To show that $\boldsymbol{\theta}^{*}=\boldsymbol{\theta}_{\text {true }}$, we note that

$$
\begin{aligned}
\frac{d}{d \boldsymbol{\theta}} \int \log f(\mathbf{x} ; \boldsymbol{\theta}) \cdot f\left(\mathbf{x} ; \boldsymbol{\theta}_{\text {true }}\right) d \mathbf{x} & =\int \frac{d}{d \boldsymbol{\theta}} \log f(\mathbf{x} ; \boldsymbol{\theta}) \cdot f\left(\mathbf{x} ; \boldsymbol{\theta}_{\text {true }}\right) d \mathbf{x} \\
& =\int \frac{f^{\prime}(\mathbf{x} ; \boldsymbol{\theta})}{f(\mathbf{x} ; \boldsymbol{\theta})} \cdot f\left(\mathbf{x} ; \boldsymbol{\theta}_{\text {true }}\right) d \mathbf{x}
\end{aligned}
$$

We ask whether this is equal to zero. Putting $\boldsymbol{\theta}=\boldsymbol{\theta}_{\text {true }}$, we have that

$$
\int \frac{f^{\prime}\left(\mathbf{x} ; \boldsymbol{\theta}_{\text {true }}\right)}{f\left(\mathbf{x} ; \boldsymbol{\theta}_{\text {true }}\right)} \cdot f\left(\mathbf{x} ; \boldsymbol{\theta}_{\text {true }}\right) d \mathbf{x}=\int f^{\prime}\left(\mathbf{x} ; \boldsymbol{\theta}_{\text {true }}\right) d \mathbf{x} .
$$

However, this integral can be simplified to

$$
\int f^{\prime}\left(\mathbf{x} ; \boldsymbol{\theta}_{\text {true }}\right) d \mathbf{x}=\left.\frac{d}{d \boldsymbol{\theta}} \underbrace{\int f(\mathbf{x} ; \boldsymbol{\theta}) d \mathbf{x}}_{=1}\right|_{\boldsymbol{\theta}=\boldsymbol{\theta}_{\text {true }}}=0 .
$$

Therefore, $\boldsymbol{\theta}_{\text {true }}$ is the maximizer for $g(\boldsymbol{\theta})$, and so $\boldsymbol{\theta}_{\text {true }}=\boldsymbol{\theta}^{*}$.

End of the discussion. Please join us again.

\subsubsection{Invariance principle}

Another useful property satisfied by the ML estimate is the invariance principle. The invariance principle says that a monotonic transformation of the true parameter is preserved for the ML estimates.

What is the invariance principle?

- There is a monotonic function $h$.

- There is an ML estimate $\widehat{\theta}$ for $\theta$.

- The monotonic function $h$ maps the true parameter $\theta \longmapsto h(\theta)$.

- Then the same function will map the ML estimate $\widehat{\theta} \longmapsto h\left(\widehat{\theta}\right)$.

The formal statement of the invariance principle is given by the theorem below.

Theorem 8.1. If $\widehat{\theta}_{M L}$ is the $M L$ estimate of $\theta$, then for any one-to-one function $h$ of $\theta$, the $M L$ estimate of $h(\theta)$ is $h\left(\widehat{\theta}_{M L}\right)$.

Proof. Define the likelihood function $\mathcal{L}(\theta)$ (we have dropped $\mathbf{x}$ to simplify the notation). Then, for any monotonic function $h$, we have that

$$
\mathcal{L}(\theta)=\mathcal{L}\left(h^{-1}(h(\theta))\right)
$$



\subsection{PROPERTIES OF ML ESTIMATES}

Let $\widehat{\theta}$ be the ML estimate:

$$
\widehat{\theta}=\underset{\theta}{\operatorname{argmax}} \mathcal{L}(\theta)=\underset{\theta}{\operatorname{argmax}} \mathcal{L}\left(h^{-1}(h(\theta))\right) .
$$

By the definition of ML, $\widehat{\theta}$ must maximize the likelihood. Therefore, $\mathcal{L}\left(h^{-1}(h(\theta))\right)$ is maximized when $h^{-1}(h(\theta))=\widehat{\theta}$. This implies that $h(\theta)=h\left(\widehat{\theta}\right)$ because $h$ is monotonic. Since $h(\theta)$ is the parameter we try to estimate, the equality $h(\theta)=h\left(\widehat{\theta}\right)$ implies that $h\left(\widehat{\theta}\right)$ is the ML estimate of $h(\theta)$.

Example 8.18. Consider the single-photon image sensor example we discussed in Section 8.1. We consider a set of i.i.d. Bernoulli random variables with PMF

$$
p_{x^{(n)}}(1)=1-e^{-\eta} \quad \text { and } \quad p_{x^{(n)}}(0)=e^{-\eta}
$$

Find the ML estimate through (a) direct calculation and (b) the invariance principle.

Solution. (a) Following the example in Equation 8.12), the ML estimate of $\eta$ is

$$
\widehat{\eta}_{\mathrm{ML}}=\underset{\eta}{\operatorname{argmax}} \prod_{n=1}^{N}\left(1-e^{-\eta}\right)^{x^{(n)}}\left(e^{-\eta}\right)^{1-x^{(n)}}=-\log \left(1-\frac{1}{N} \sum_{n=1}^{N} x^{(n)}\right)
$$

(b) We can obtain the same result using the invariance principle. Since $x^{(n)}$ is a binary random variable, we assume that it is a Bernoulli with parameter $\theta$. Then the ML estimate of $\theta$ is

$$
\begin{aligned}
\widehat{\theta} & =\underset{\theta}{\operatorname{argmax}} \prod_{n=1}^{N} \theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}} \\
& =\frac{1}{N} \sum_{n=1}^{N} x^{(n)} .
\end{aligned}
$$

The relationship between $\theta$ and $\eta$ is that $\theta=1-e^{-\eta}$, or $\eta=-\log (1-\theta)$. So we let $h(\theta)=-\log (1-\theta)$. The invariance principle says that the ML estimate of $h(\theta)$ is

$$
\begin{aligned}
\widehat{\eta}_{\mathrm{ML}} \stackrel{\text { def }}{=} \widehat{h(\theta)_{\mathrm{ML}}} & \stackrel{(i)}{=} h\left(\widehat{\theta}\right) \\
& =-\log \left(1-\frac{1}{N} \sum_{n=1}^{N} x^{(n)}\right),
\end{aligned}
$$

where (i) follows from the invariance principle.

The invariance principle can be very convenient, especially when the transformation $h$ is complicated, so that a direct evaluation of the ML estimate is difficult.

The invariance principle is portrayed in Figure 8.14. We start with the Bernoulli loglikelihood

$$
\log \mathcal{L}(\theta \mid S)=S \log \theta+(1-S) \log (1-\theta)
$$



\section{CHAPTER 8. ESTIMATION}
![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-518.jpg?height=812&width=1024&top_left_y=233&top_left_x=328)

Figure 8.14: The invariance principle is a transformation of the $M L$ estimate. In this example, we consider a Bernoulli log-likelihood function shown in the lowermost plot. For this log-likelihood, the ML estimate is $\widehat{\theta}=0.4$. On the left-hand side we show another log-likelihood, derived for a truncated Poisson random variable. Note that the $\mathrm{ML}$ estimate is $\widehat{\eta}_{\mathrm{ML}}=0.5108$. The invariance principle asserts that, instead of computing these $\mathrm{ML}$ estimates directly, we can first derive the relationship between $\eta$ and $\theta$ for any $\theta$. Since we know that $\theta=1-e^{-\eta}$, it follows that $\eta=-\log (1-\theta)$. We define this transformation as $\eta=h(\theta)=-\log (1-\theta)$. Then the $\mathrm{ML}$ estimate is $\widehat{\eta}_{\mathrm{ML}}=h\left(\widehat{\theta}\right)=h(0.4)=0.5108$. The invariance principle saves us the trouble of computing the maximization of the more truncated Poisson likelihood.

In this particular example we let $S=20$, where $S$ denotes the sum of the $N=50$ Bernoulli random variables. The other log-likelihood is the truncated Poisson, which is given by

$$
\log \mathcal{L}(\eta \mid S)=S \log \left(1-e^{-\eta}\right)+(1-S) \log \left(e^{-\eta}\right)
$$

The transformation between the two is the function $\eta=h(\theta)=-\log (1-\theta)$. Putting everything into the figure, we see that the ML estimate $(\theta=0.4)$ is translated to $\eta=0.5108$. The invariance principle asserts that this calculation can be done by $\widehat{\eta}_{\mathrm{ML}}=h\left(\widehat{\theta}\right)=$ $h(0.4)=-0.5108$.

\subsection{Maximum A Posteriori Estimation}

In ML estimation, the parameter $\boldsymbol{\theta}$ is treated as a deterministic quantity. There are, however, many situations where we have some prior knowledge about $\boldsymbol{\theta}$. For example, we may not know exactly the speed of a car, but we may know that the speed is roughly $65 \mathrm{mph}$

\subsection{MAXIMUM A POSTERIORI ESTIMATION}

with a standard deviation of $5 \mathrm{mph}$. How do we incorporate such prior knowledge into the estimation problem?

In this section, we introduce the second estimation technique, known as the maximum a posteriori (MAP) estimation. MAP estimation links the likelihood and the prior. The key idea is to treat the parameter $\boldsymbol{\theta}$ as a random variable (vector) $\Theta$ with a $\operatorname{PDF} f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})$.

\subsubsection{The trio of likelihood, prior, and posterior}

To understand how the MAP estimation works, it is important first to understand the role of the parameter $\boldsymbol{\theta}$, which changes from a deterministic quantity to a random quantity.

Recall the likelihood function we defined in the ML estimation; it is

$$
\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x})=f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})
$$

if we assume that we have a set of i.i.d. observations $\mathbf{x}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$. By writing the PDF of $\mathbf{X}$ as $f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$, we emphasize that $\boldsymbol{\theta}$ is a deterministic but unknown parameter. There is nothing random about $\boldsymbol{\theta}$.

In MAP, we change the nature of $\boldsymbol{\theta}$ from deterministic to random. We replace $\boldsymbol{\theta}$ by $\boldsymbol{\Theta}$ and write

$$
f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta}) \stackrel{\text { becomes }}{\Longrightarrow} f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})
$$

The difference between the left-hand side and the right-hand side is subtle but important. On the left-hand side, $f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$ is the PDF of $\mathbf{X}$. This PDF is parameterized by $\boldsymbol{\theta}$. On the right-hand side, $f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})$ is a conditional PDF of $\mathbf{X}$ given $\boldsymbol{\Theta}$. The values they provide are exactly the same. However, in $f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta}), \boldsymbol{\theta}$ is a realization of a random variable $\boldsymbol{\Theta}$.

Because $\boldsymbol{\Theta}$ is now a random variable (vector), we can define its PDF (yes, the PDF of $\Theta)$, and denote it by

$$
f_{\Theta}(\boldsymbol{\theta})
$$

which is called the prior distribution. The prior distribution of $\Theta$ is unique in MAP estimation. There is nothing called a prior in ML estimation.

Multiplying $f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})$ with the prior $\operatorname{PDF} f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})$, and using Bayes' Theorem, we obtain the posterior distribution:

$$
f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})=\frac{f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta}) f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})}{f_{\mathbf{X}}(\mathbf{x})}
$$

The posterior distribution is the $\mathrm{PDF}$ of $\boldsymbol{\Theta}$ given the measurements $\mathbf{X}$.

The likelihood, the prior, and the posterior can be confusing. Let us clarify their meanings.

- Likelihood $f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})$ : This is the conditional probability density of $\mathbf{X}$ given the parameter $\boldsymbol{\Theta}$. Do not confuse the likelihood $f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})$ defined in the MAP context and the likelihood $f_{\mathbf{X}}(\mathbf{x} ; \mid \boldsymbol{\theta})$ defined in the ML context. The former assumes that $\boldsymbol{\Theta}$ is random whereas the latter assumes that $\boldsymbol{\theta}$ is deterministic. They have the same values.

- Prior $f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})$ : This is the prior distribution of $\boldsymbol{\Theta}$. It does not come from the data $\mathbf{X}$ but from our prior knowledge. For example, if we see a bike on the road, even before we take any measurement we will have a rough idea of its speed. This is the prior distribution.

\section{CHAPTER 8. ESTIMATION}

- Posterior $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$ : This is the posterior density of $\boldsymbol{\Theta}$ given that we have observed $\mathbf{X}$. Do not confuse $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$ and $\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x})$. The posterior distribution $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$ is a PDF of $\boldsymbol{\Theta}$ given $\mathbf{X}=\mathbf{x}$. The likelihood $\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x})$ is not a PDF. If you integrate $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$ with respect to $\boldsymbol{\theta}$, you get 1 , but if you integrate $\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x})$ with respect to $\boldsymbol{\theta}$, you do not get 1 .

What is the difference between ML and MAP?

Likelihood ML $f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$ The parameter $\boldsymbol{\theta}$ is deterministic.

MAP $f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})$ The parameter $\boldsymbol{\Theta}$ is random.

Prior ML There is no prior, because $\boldsymbol{\theta}$ is deterministic.

MAP $f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})$ This is the PDF of $\boldsymbol{\Theta}$.

Optimization ML Find the peak of the likelihood $f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$.

MAP Find the peak of the posterior $f_{\Theta \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$.

Maximum a posteriori (MAP) estimation is a form of Bayesian estimation. Bayesian methods emphasize our prior knowledge or beliefs about the parameters. As we will see shortly, the prior has something valuable to offer, especially when we have very few data points.

\subsubsection{Understanding the priors}

Since the biggest difference between MAP and ML is the addition of the prior $f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})$, we need to take a closer look at what they mean. In Figure 8.15 below, we show a set of six different priors. We ask two questions: (1) What do they mean? (2) Which one should we use?
![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-520.jpg?height=244&width=1354&top_left_y=1696&top_left_x=150)

Figure 8.15: This figure illustrates six different examples of the prior distribution $f_{\Theta}(\theta)$, when the prior is a 1D parameter $\theta$. The prior distribution $f_{\Theta}(\theta)$ is the PDF of $\Theta$. (a) $f_{\Theta}(\theta)=\delta(\theta)$, which is a delta function. (b) $f_{\Theta}(\theta)=\frac{1}{b-a}$ for $a \leq \theta \leq b$. This is a uniform distribution. (c) This is also a uniform distribution, but the spread is very wide. (d) $f_{\Theta}(\theta)=$ Gaussian $\left(0, \sigma^{2}\right)$, which is a zero-mean Gaussian. (e) The same Gaussian, but with a different mean. (f) A Gaussian with zero mean, but a large variance. What does the shape of a prior tell us?

It tells us your belief as to how the underlying parameter $\boldsymbol{\Theta}$ should be distributed.

The meaning of this statement can be best understood from the examples shown in Figure 8.15

- Figure 8.15(a). This is a delta prior $f_{\Theta}(\theta)=\delta(\theta)$ (or $f_{\Theta}(\theta)=\delta\left(\theta-\theta_{0}\right)$ ). If you use this prior, you are absolutely sure that the parameter $\Theta$ takes a specific value. There is no uncertainty about your belief. Since you are so confident about your prior knowledge, you will ignore the likelihood that is constructed from the data. No one will use a delta prior in practice.

- Figure 8.15(b). $f_{\Theta}(\theta)=\frac{1}{b-a}$ for $a \leq \theta \leq b$, and is zero otherwise. This is a bounded uniform prior. You do not have any preference for the parameter $\Theta$, but you do know from your prior experience that $a \leq \Theta \leq b$.

- Figure 8.15 (c). This prior is the same as (b) but is short and very wide. If you use this prior, it means that you know nothing about the parameter. So you give up the prior and let the likelihood dominate the MAP estimate.

- Figure 8.15(d). $f_{\Theta}(\theta)=$ Gaussian $\left(0, \sigma^{2}\right)$. You use this prior when you know something about the parameter, e.g., that it is centered at certain location and you have some uncertainty.

- Figure 8.15(e). Same as (d), but the parameter is centered at some other location.

- Figure 8.15(f). Same as (d), but you have less confidence about the parameter.

As you can see from these examples, the shape of the prior tells us how you want $\Theta$ to be distributed. The choice you make will directly influence the MAP optimization, and hence the MAP estimate.

Since the prior is a subjective quantity in the MAP framework, you as the user have the freedom to choose whatever you like. For instance, if you have conducted a similar experiment before, you can use the results of the previous experiments as the current prior. Another strategy is to go with physics. For instance, we can argue that $\boldsymbol{\theta}$ should be sparse so that it contains as few non-zeros as possible. In this case, a sparsity-driven prior, such as $f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})=\exp \left\{-\|\boldsymbol{\theta}\|_{1}\right\}$, could be a choice. The third strategy is to choose a prior that is computationally "friendlier", e.g., in quadratic form so that the MAP is differentiable. One such choice is the conjugate prior. We will discuss this later in Section 8.3.6.

Which prior should we choose?

- Based on your preference, e.g., you know from historical data that the parameter should behave in certain ways.

- Based on physics, e.g., the parameter has a physical interpretation, so you need to abide by the physical laws.

- Choose a prior that is computationally "friendlier". This is the topic of the conjugate prior, which is a prior that does not change the form of the posterior distribution. (We will discuss this later in Section 8.3.6.)

\section{CHAPTER 8. ESTIMATION}

\subsubsection{MAP formulation and solution}

Our next task is to study how to formulate the MAP problem and how to solve it.

Definition 8.7. Let $\mathbf{X}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$ be i.i.d. observations. Let $\boldsymbol{\Theta}$ be a random parameter. The maximum-a-posteriori estimate of $\boldsymbol{\Theta}$ is

$$
\widehat{\theta}_{M A P}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}} f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})
$$

Philosophically speaking, ML and MAP have two different goals. ML considers a parametric model with a deterministic parameter. Its goal is to find the parameter that maximizes the likelihood for the data we have observed. MAP also considers a parametric model but the parameter $\boldsymbol{\Theta}$ is random. Because $\boldsymbol{\Theta}$ is random, we are finding one particular state $\boldsymbol{\theta}$ of the parameter $\boldsymbol{\Theta}$ that offers the best explanation conditioned on the data $\mathbf{X}$ we observe. In a sense, the two optimization problems are

$$
\begin{aligned}
& \widehat{\boldsymbol{\theta}}_{\mathrm{ML}}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}} f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta}), \\
& \widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}} f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x}) .
\end{aligned}
$$

This pair of equations is interesting, as the pair tells us that the difference between the ML estimation and the MAP estimation is the flipped order of $\mathbf{X}$ and $\boldsymbol{\Theta}$.

There are two reasons we care about the posterior. First, in MAP the posterior allows us to incorporate the prior. ML does not allow a prior. A prior can be useful when the number of samples is small. Second, maximizing the posterior does have some physical interpretations. MAP asks for the probability of $\boldsymbol{\Theta}=\boldsymbol{\theta}$ after observing $N$ training samples $\mathbf{X}=\mathbf{x}$. ML asks for the probability of observing $\mathbf{X}=\mathbf{x}$ given a parameter $\boldsymbol{\theta}$. Both are correct and legitimate criteria, but sometimes we might prefer one over the other.

To solve the MAP problem, we notice that

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}} & =\underset{\boldsymbol{\theta}}{\operatorname{argmax}} f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x}) \\
& =\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \frac{f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta}) f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})}{f_{\mathbf{X}}(\mathbf{x})} \\
& =\underset{\boldsymbol{\theta}}{\operatorname{argmax}} f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta}) f_{\boldsymbol{\Theta}}(\boldsymbol{\theta}), \quad f_{\mathbf{X}}(\mathbf{x}) \text { does not contain } \boldsymbol{\theta} \\
& =\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \log f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})+\log f_{\boldsymbol{\Theta}}(\boldsymbol{\theta}) .
\end{aligned}
$$

Therefore, what MAP adds is the prior $\log f_{\Theta}(\boldsymbol{\theta})$. If you use an uninformative prior, e.g., a prior with extremely wide support, then the MAP estimation will return more or less the same result as the ML estimation.

When does MAP $=$ ML?

- The relation "=" does not make sense here, because $\boldsymbol{\theta}$ is random in MAP but deterministic in ML.

- Solution of MAP optimization $=$ solution of $M L$ optimization, when $f_{\Theta}(\boldsymbol{\theta})$ is uniform over the parameter space.

\subsection{MAXIMUM A POSTERIORI ESTIMATION}

- In this case, $f_{\Theta}(\boldsymbol{\theta})=$ constant and so it can be dropped from the optimization.

Example 8.19. Let $x^{(1)}, \ldots, x^{(n)}$ be i.i.d. random variables with a $\operatorname{PDF} f_{x^{(n)} \mid \Theta}\left(x^{(n)} \mid \theta\right)$ for all $n$, and $\Theta$ be a random parameter with $\operatorname{PDF} f_{\Theta}(\theta)$ :

$$
\begin{aligned}
f_{x^{(n)} \mid \Theta}\left(x^{(n)} \mid \theta\right) & =\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}\right\}, \\
f_{\Theta}(\theta) & =\frac{1}{\sqrt{2 \pi \sigma_{0}^{2}}} \exp \left\{-\frac{\left(\theta-\mu_{0}\right)^{2}}{2 \sigma_{0}^{2}}\right\} .
\end{aligned}
$$

Find the MAP estimate.

Solution. The MAP estimate is

$$
\begin{aligned}
& \widehat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\operatorname{argmax}}\left[\prod_{n=1}^{N} \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}\right\}\right] \times\left[\frac{1}{\sqrt{2 \pi \sigma_{0}^{2}}} \exp \left\{-\frac{\left(\theta-\mu_{0}\right)^{2}}{2 \sigma_{0}^{2}}\right\}\right] \\
& =\underset{\theta}{\operatorname{argmax}}\left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)^{N} \times \frac{1}{\sqrt{2 \pi \sigma_{0}^{2}}} \exp \left\{-\sum_{n=1}^{N} \frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}-\frac{\left(\theta-\mu_{0}\right)^{2}}{2 \sigma_{0}^{2}}\right\} .
\end{aligned}
$$

Since the maximizer is not changed by any monotonic function, we apply logarithm to the above equations. This yields

$$
\begin{aligned}
\widehat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\operatorname{argmax}} & \left\{-\frac{N}{2} \log \left(2 \pi \sigma^{2}\right)-\frac{1}{2} \log \left(2 \pi \sigma_{0}^{2}\right)\right. \\
& \left.-\sum_{n=1}^{N} \frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}-\frac{\left(\theta-\mu_{0}\right)^{2}}{2 \sigma_{0}^{2}}\right\} .
\end{aligned}
$$

Constants in the maximization do not matter. So by dropping the constant terms we obtain

$$
\widehat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\operatorname{argmax}}\left\{-\sum_{n=1}^{N} \frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}-\frac{\left(\theta-\mu_{0}\right)^{2}}{2 \sigma_{0}^{2}}\right\} .
$$

It now remains to solve the maximization. To this end we take the derivative w.r.t. $\theta$ and show that

$$
\frac{d}{d \theta}\left\{-\sum_{n=1}^{N} \frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}-\frac{\left(\theta-\mu_{0}\right)^{2}}{2 \sigma_{0}^{2}}\right\}=0 .
$$

This yields

$$
\sum_{n=1}^{N} \frac{\left(x^{(n)}-\theta\right)}{\sigma^{2}}-\frac{\theta-\mu_{0}}{\sigma_{0}^{2}}=0
$$



\section{CHAPTER 8. ESTIMATION}

Rearranging the terms gives us the final result:

$$
\widehat{\theta}_{\mathrm{MAP}}=\frac{\sigma_{0}^{2}\left(\frac{1}{N} \sum_{n=1}^{N} x^{(n)}\right)+\frac{\sigma^{2}}{N} \mu_{0}}{\sigma_{0}^{2}+\frac{\sigma^{2}}{N}}
$$

Practice Exercise 8.7. Prove that if $f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})=\delta\left(\boldsymbol{\theta}-\boldsymbol{\theta}_{0}\right)$, the MAP estimate is $\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}=$ $\boldsymbol{\theta}_{0}$

Solution. If $f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})=\delta\left(\boldsymbol{\theta}-\boldsymbol{\theta}_{0}\right)$, then

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}} & =\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \log f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})+\log f_{\boldsymbol{\Theta}}(\boldsymbol{\theta}) \\
& =\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \log f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})+\log \delta\left(\boldsymbol{\theta}-\boldsymbol{\theta}_{0}\right) \\
& =\left\{\begin{array}{lll}
\underset{\boldsymbol{\theta}}{\operatorname{argmax}} & \log f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})-\infty, & \boldsymbol{\theta} \neq \boldsymbol{\theta}_{0} . \\
\underset{\boldsymbol{\theta}}{\operatorname{argmax}} & \log f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})+0, & \boldsymbol{\theta}=\boldsymbol{\theta}_{0} .
\end{array}\right.
\end{aligned}
$$

Thus, if $\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}} \neq \boldsymbol{\theta}_{0}$, the first case says that there is no solution, so we must go with the second case $\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}=\boldsymbol{\theta}_{0}$. But if $\boldsymbol{\boldsymbol { \theta }}_{\mathrm{MAP}}=\boldsymbol{\theta}_{0}$, there is no optimization because we have already chosen $\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}=\boldsymbol{\theta}_{0}$. This proves the result.

\subsubsection{Analyzing the MAP solution}

As we said earlier, MAP offers something that ML does not. To see this, we will use the result of the Gaussian random variables as an example and analyze the MAP solution as we change the parameters $N$ and $\sigma_{0}$. Recall that if $x^{(1)}, \ldots, x^{(n)}$ are i.i.d. Gaussian random variables with unknown mean $\theta$ and known variance $\sigma$, the ML estimate is

$$
\widehat{\theta}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}
$$

Assuming that the parameter $\Theta$ is distributed according to a $\operatorname{PDF} \operatorname{Gaussian}\left(\mu_{0}, \sigma_{0}^{2}\right)$, we have shown in the previous subsection that

$$
\widehat{\theta}_{\mathrm{MAP}}=\frac{\sigma_{0}^{2}\left(\frac{1}{N} \sum_{n=1}^{N} x^{(n)}\right)+\frac{\sigma^{2}}{N} \mu_{0}}{\sigma_{0}^{2}+\frac{\sigma^{2}}{N}}=\frac{\sigma_{0}^{2} \widehat{\theta}+\frac{\sigma^{2}}{N} \mu_{0}}{\sigma_{0}^{2}+\frac{\sigma^{2}}{N}} .
$$

In what follows, we will take a look at the behavior of the MAP estimate $\widehat{\theta}_{\text {MAP }}$ as $N$ and $\sigma_{0}$ change. The results of our discussion are summarized in Figure 8.16.

First, let's look at the effect of $N$.

How does $N$ change $\widehat{\theta}_{\text {MAP }}$ ?

- As $N \rightarrow \infty$, the MAP estimate $\widehat{\theta}_{\mathrm{MAP}} \rightarrow \widehat{\theta}$ : If we have enough samples, we trust the data.

\subsection{MAXIMUM A POSTERIORI ESTIMATION}

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-525.jpg?height=351&width=654&top_left_y=237&top_left_x=249)

(a) Effect of $N$

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-525.jpg?height=349&width=554&top_left_y=236&top_left_x=1029)

(b) Effect of $\sigma_{0}$

Figure 8.16: The MAP estimate $\widehat{\theta}_{\mathrm{MAP}}$ swings between the ML estimate $\widehat{\theta}$ and the prior $\mu_{0}$. (a) When $N$ increases, the likelihood is more reliable and so we lean towards the $\mathrm{ML}$ estimate. If $N$ is small, we should trust the prior more than the ML estimate. (b) When $\sigma_{0}$ decreases, we become more confident about the prior and so we will use it. If $\sigma_{0}$ is large, we use more information from the ML estimate.

- As $N \rightarrow 0$, the MAP estimate $\widehat{\theta}_{\mathrm{MAP}} \rightarrow \theta_{0}$. If we do not have any samples, we trust the prior.

These two results can be demonstrated by taking the limits. As $N \rightarrow \infty$, the MAP estimate converges to

$$
\lim _{N \rightarrow \infty} \widehat{\theta}_{\mathrm{MAP}}=\lim _{N \rightarrow \infty} \frac{\sigma_{0}^{2} \widehat{\theta}+\frac{\sigma^{2}}{N} \mu_{0}}{\sigma_{0}^{2}+\frac{\sigma^{2}}{N}}=\widehat{\theta} .
$$

This result is not surprising. When we have infinitely many samples, we will completely rely on the data and make our estimate. Thus, the MAP estimate is the same as the ML estimate.

When $N \rightarrow 0$, the MAP estimate converges to

$$
\lim _{N \rightarrow 0} \widehat{\theta}_{\mathrm{MAP}}=\lim _{N \rightarrow 0} \frac{\sigma_{0}^{2} \widehat{\theta}+\frac{\sigma^{2}}{N} \mu_{0}}{\sigma_{0}^{2}+\frac{\sigma^{2}}{N}}=\mu_{0} .
$$

This means that, when we do not have any samples, the MAP estimate $\widehat{\theta}_{\text {MAP }}$ will completely use the prior distribution, which has a mean $\mu_{0}$.

The implication of this result is that MAP offers a natural swing between $\widehat{\theta}$ and $\widehat{\theta}_{0}$, controlled by $N$. Where does this $N$ come from? If we recall the derivation of the result, we note that the $N$ affects the likelihood term through the number of samples:

$$
\widehat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\operatorname{argmax}}\{-\underbrace{\sum_{n=1}^{N} \frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}}_{N \text { terms here }}-\underbrace{\frac{\left(\theta-\mu_{0}\right)^{2}}{2 \sigma_{0}^{2}}}_{1 \text { term }}\} .
$$

Thus, as $N$ increases, the influence of the data term grows, and so the result will gradually shift towards $\widehat{\theta}$.

Figure 8.17 illustrates a numerical experiment in which we draw $N$ random samples $x^{(1)}, \ldots, x^{(n)}$ according to a Gaussian distribution $\operatorname{Gaussian}\left(\theta, \sigma^{2}\right)$, with $\sigma=1$. We assume that the prior distribution is $\operatorname{Gaussian}\left(\mu_{0}, \sigma_{0}^{2}\right)$, with $\mu_{0}=0$ and $\sigma_{0}=0.25$. The ML estimate of this problem is $\widehat{\theta}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}$, whereas the MAP estimate is given by Equation 8.40 .

\section{CHAPTER 8. ESTIMATION}

The figure shows the resulting PDFs. A helpful analogy is that the prior and the likelihood are pulling a rope in two opposite directions. As $N$ grows, the force of the likelihood increases and so the influence becomes stronger.

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-526.jpg?height=486&width=678&top_left_y=450&top_left_x=164)

(a) $N=1$

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-526.jpg?height=518&width=678&top_left_y=415&top_left_x=867)

(b) $N=50$

Figure 8.17: The subfigures show the prior distribution $f_{\Theta}(\theta)$ and the likelihood function $f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta)$, given the observed data. (a) When $N=1$, the estimated posterior distribution $f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x})$ is pulled towards the prior. (b) When $N=50$, the posterior is pulled towards the $\mathrm{ML}$ estimate. The analogy for the situation is that each data point is acting as a small force against the big force of the prior. As $N$ grows, the small forces of the data points accumulate and eventually dominate.

We next look at the effect of $\sigma_{0}$.

How does $\sigma_{0}$ change $\widehat{\theta}_{\text {MAP }}$ ?

- As $\sigma_{0} \rightarrow \infty$, the MAP estimate $\widehat{\theta}_{\mathrm{MAP}} \rightarrow \widehat{\theta}$ : If we have doubts about the prior, we trust the data.

- As $\sigma_{0} \rightarrow 0$, the MAP estimate $\widehat{\theta}_{\mathrm{MAP}} \rightarrow \theta_{0}$. If we are absolutely sure about the prior, we ignore the data.

When $\sigma_{0} \rightarrow \infty$, the limit of $\widehat{\theta}_{\mathrm{MAP}}$ is

$$
\lim _{\sigma_{0} \rightarrow \infty} \widehat{\theta}_{\mathrm{MAP}}=\lim _{\sigma_{0} \rightarrow \infty} \frac{\sigma_{0}^{2} \widehat{\theta}+\frac{\sigma^{2}}{N} \mu_{0}}{\sigma_{0}^{2}+\frac{\sigma^{2}}{N}}=\widehat{\theta} .
$$

The reason why this happens is that $\sigma_{0}$ is the uncertainty level of the prior. If $\sigma_{0}$ is high, we are not certain about the prior. In this case, MAP chooses to follow the ML estimate.

When $\sigma_{0} \rightarrow 0$, the limit of $\widehat{\theta}_{\mathrm{MAP}}$ is

$$
\lim _{\sigma_{0} \rightarrow 0} \widehat{\theta}_{\mathrm{MAP}}=\lim _{\sigma_{0} \rightarrow 0} \frac{\sigma_{0}^{2} \widehat{\theta}+\frac{\sigma^{2}}{N} \mu_{0}}{\sigma_{0}^{2}+\frac{\sigma^{2}}{N}}=\mu_{0} .
$$

Note that when $\sigma_{0} \rightarrow 0$, we are essentially saying that we are absolutely sure about the prior. If we are so sure about the prior, there is no need to look at the data. In that case the MAP estimate is $\mu_{0}$.

\subsection{MAXIMUM A POSTERIORI ESTIMATION}

The way to understand the influence of $\sigma_{0}$ is to inspect the equation:

$$
\widehat{\theta}_{\text {MAP }}=\underset{\theta}{\operatorname{argmax}}\{-\underbrace{\sum_{n=1}^{N} \frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}}_{\text {fixed w.r.t. } \sigma_{0}}-\underbrace{\frac{\left(\theta-\mu_{0}\right)^{2}}{2 \sigma_{0}^{2}}}_{\text {changes with } \sigma_{0}}\} .
$$

Since $\sigma_{0}$ is purely a preference you decide, you can control how much trust to put onto the prior.

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-527.jpg?height=478&width=669&top_left_y=647&top_left_x=229)

(a) $\sigma_{0}=0.1$

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-527.jpg?height=478&width=666&top_left_y=647&top_left_x=917)

(b) $\sigma_{0}=1$

Figure 8.18: The subfigures show the prior distribution $f_{\Theta}(\theta)$ and the likelihood function $f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta)$, given the observed data. (a) When $\sigma_{0}=0.1$, the estimated posterior distribution $f_{\Theta \mid X}(\theta \mid \mathbf{x})$ is pulled towards the prior. (b) When $\sigma_{0}=1$, the posterior is pulled towards the $\mathrm{ML}$ estimate. An analogy for the situation is that the strength of the prior depends on the magnitude of $\sigma_{0}$. If $\sigma_{0}$ is small the prior is strong, and so the influence is large. If $\sigma_{0}$ is large the prior is weak, and so the $\mathrm{ML}$ estimate will dominate.

Figure 8.18 illustrates a numerical experiment in which we compare $\sigma_{0}=0.1$ and $\sigma_{0}=1$. If $\sigma_{0}$ is small, the prior distribution $f_{\Theta}(\theta)$ becomes similar to a delta function. We can interpret it as a very confident prior, so confident that we wish to align with the prior. The situation can be imagined as a game of tug-of-war between a powerful bull and a horse, which the bull will naturally win. If $\sigma_{0}$ is large the prior distribution will become flat. It means that we are not very confident about the prior so that we will trust the data. In this case the MAP estimate will shift towards the ML estimate.

\subsubsection{Analysis of the posterior distribution}

When the likelihood is multiplied with the prior to form the posterior, what does the posterior distribution look like? To answer this question we continue our Gaussian example with a fixed variance $\sigma$ and an unknown mean $\theta$. The posterior distribution is proportional to

$$
\begin{aligned}
f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x}) & =\frac{f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta) f_{\Theta}(\theta)}{f_{\mathbf{X}}(\mathbf{x})} \propto f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta) f_{\Theta}(\theta) \\
& =\left[\prod_{n=1}^{N} \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}\right\}\right] \cdot\left[\frac{1}{\sqrt{2 \pi \sigma_{0}^{2}}} \exp \left\{-\frac{\left(\theta-\mu_{0}\right)^{2}}{2 \sigma_{0}^{2}}\right\}\right]
\end{aligned}
$$



\section{CHAPTER 8. ESTIMATION}

Performing the multiplication and completing the squares,

$$
\sum_{n=1}^{N} \frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}+\frac{\left(\theta-\mu_{0}\right)^{2}}{2 \sigma_{0}^{2}}=\frac{\left(\theta-\widehat{\theta}_{\mathrm{MAP}}\right)^{2}}{2 \sigma_{\mathrm{MAP}}^{2}}
$$

where

$$
\widehat{\theta}_{\mathrm{MAP}}=\frac{\sigma_{0}^{2} \widehat{\theta}+\frac{\sigma^{2}}{N} \mu_{0}}{\sigma_{0}^{2}+\frac{\sigma^{2}}{N}}, \quad \text { and } \quad \frac{1}{\widehat{\sigma}_{\mathrm{MAP}}^{2}}=\frac{1}{\sigma_{0}^{2}}+\frac{N}{\sigma^{2}}
$$

In other words, the posterior distribution $f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x})$ is also a Gaussian with

$$
f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x})=\operatorname{Gaussian}\left(\widehat{\theta}_{\mathrm{MAP}}, \widehat{\sigma}_{\mathrm{MAP}}^{2}\right)
$$

If $f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})=\operatorname{Gaussian}(\mathbf{x} ; \theta, \sigma)$, and $f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})=\operatorname{Gaussian}\left(\theta ; \mu_{0}, \sigma_{0}^{2}\right)$, what is the posterior $f_{\Theta \mid X}(\theta \mid x)$ ?

The posterior $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$ is $\operatorname{Gaussian}\left(\widehat{\theta}_{\mathrm{MAP}}, \widehat{\sigma}_{\mathrm{MAP}}^{2}\right)$, where

$$
\widehat{\theta}_{\mathrm{MAP}}=\frac{\sigma_{0}^{2} \widehat{\theta}+\frac{\sigma^{2}}{N} \mu_{0}}{\sigma_{0}^{2}+\frac{\sigma^{2}}{N}}, \quad \text { and } \quad \frac{1}{\widehat{\sigma}_{\mathrm{MAP}}^{2}}=\frac{1}{\sigma_{0}^{2}}+\frac{N}{\sigma^{2}} .
$$

The posterior tells us how $N$ and $\sigma_{0}$ will influence the MAP estimate. As $N$ grows, the posterior mean and variance becomes

$$
\lim _{N \rightarrow \infty} \widehat{\theta}_{\mathrm{MAP}}=\widehat{\theta}=\theta, \quad \text { and } \quad \lim _{N \rightarrow \infty} \widehat{\sigma}_{\mathrm{MAP}}=0 .
$$

As a result, the posterior distribution $f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x})$ will converge to a delta function centered at the ML estimate $\widehat{\theta}$. Therefore, as we try to solve the MAP problem by maximizing the posterior, the MAP estimate has to improve because $\widehat{\sigma}_{\mathrm{MAP}} \rightarrow 0$.

We can plot the posterior distribution Gaussian $\left(\hat{\theta}_{\mathrm{MAP}}, \widehat{\sigma}_{\mathrm{MAP}}^{2}\right)$ as a function of the number of samples $N$. Figure 8.19 illustrates this example using the following configurations. The likelihood is Gaussian with $\mu=1, \sigma=0.25$. The prior is Gaussian with $\mu_{0}=0$ and $\sigma=0.25$. We construct the Gaussian according to Gaussian $\left(\widehat{\theta}_{\mathrm{MAP}}, \widehat{\sigma}_{\mathrm{MAP}}^{2}\right)$ by varying $N$. The result shown in Figure 8.19 confirms our prediction: As $N$ grows, the posterior becomes more like a delta function whose mean is the true mean $\mu$. The posterior estimator $\widehat{\theta}_{\mathrm{MAP}}$, for each $N$, is the peak of the respective Gaussian.

What is the pictorial interpretation of the MAP estimate?

- For every $N$, MAP has a posterior distribution $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$.

- As $N$ grows, $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$ converges to a delta function centered at $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}$.

- MAP tries to find the peak of $f_{\Theta \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$. For large $N$, it returns $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}$.

\subsection{MAXIMUM A POSTERIORI ESTIMATION}

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-529.jpg?height=520&width=910&top_left_y=238&top_left_x=441)

Figure 8.19: Posterior distribution $f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x})=\operatorname{Gaussian}\left(\widehat{\theta}_{\mathrm{MAP}}, \sigma_{\mathrm{MAP}}^{2}\right)$ as $N$ grows. When $N$ is small, the posterior distribution is dominated by the prior. As $N$ increases, the posterior distribution changes its mean and its variance.

\subsubsection{Conjugate prior}

Choosing the prior is an important topic in a MAP estimation. We have elaborated two "engineering" solutions: Use your prior experience or follow the physics. In this subsection, we discuss the third option: to choose something computationally friendly. To explain what we mean by "computationally friendly", let us consider the following example, thanks to Avinash Kak 4

Consider a Bernoulli distribution with a PDF

$$
f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta)=\prod_{n=1}^{N} \theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}
$$

To compute the MAP estimate, we assume that we have a prior $f_{\Theta}(\theta)$. Therefore, the MAP estimate is given by

$$
\begin{aligned}
\widehat{\theta}_{\mathrm{MAP}} & =\underset{\theta}{\operatorname{argmax}} f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta) f_{\Theta}(\theta) \\
& =\underset{\theta}{\operatorname{argmax}}\left[\prod_{n=1}^{N} \theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}\right] \cdot f_{\Theta}(\theta) \\
& =\underset{\theta}{\operatorname{argmax}} \sum_{n=1}^{N} x^{(n)} \log \theta+\left(1-x^{(n)}\right) \log (1-\theta)+\log f_{\Theta}(\theta) .
\end{aligned}
$$

Let us consider three options for the prior. Which one would you use?

- Candidate 1: $f_{\Theta}(\theta)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{(\theta-\mu)^{2}}{2 \sigma^{2}}\right\}$, a Gaussian prior. If you choose this prior, the optimization problem will become

$$
\widehat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\operatorname{argmax}} \sum_{n=1}^{N}\left\{x^{(n)} \log \theta+\left(1-x^{(n)}\right) \log (1-\theta)\right\}-\frac{(\theta-\mu)^{2}}{2 \sigma^{2}} .
$$

${ }^{4}$ Avinash Kak "ML, MAP, and Bayesian - The Holy Trinity of Parameter Estimation and Data Prediction", https://engineering.purdue.edu/kak/Tutorials/Trinity.pdf

\section{CHAPTER 8. ESTIMATION}

We can still take the derivative and set it to zero. This gives

$$
\frac{\sum_{n=1}^{N} x^{(n)}}{\theta}-\frac{N-\sum_{n=1}^{N} x^{(n)}}{1-\theta}=\frac{\theta-\mu}{\sigma^{2}} .
$$

Defining $S=\sum_{n=1}^{N} x^{(n)}$ and moving the terms around, we have

$$
(1-\theta) \sigma^{2} S-\theta \sigma^{2}(N-S)=\theta(1-\theta)(\theta-\mu) .
$$

This is a cubic polynomial problem that has a closed-form solution and is also solvable by a computer. But it's also tedious, at least to lazy engineers like ourselves.

- Candidate 2: $f_{\Theta}(\theta)=\frac{\lambda}{2} e^{-\lambda|\theta|}$, a Laplace prior. In this case, the optimization problem becomes

$$
\widehat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\operatorname{argmax}} \sum_{n=1}^{N}\left\{x^{(n)} \log \theta+\left(1-x^{(n)}\right) \log (1-\theta)\right\}-\lambda|\theta| .
$$

Welcome to convex optimization! There is no closed-form solution. If you want to solve this problem, you need to call a convex solver.

- Candidate 3: $f_{\Theta}(\theta)=\frac{1}{C} \theta^{\alpha-1}(1-\theta)^{\beta-1}$, a beta prior. This prior looks very complicated, but let's plug it into our optimization problem:

$$
\begin{aligned}
\widehat{\theta}_{\mathrm{MAP}}= & \underset{\theta}{\operatorname{argmax}} \sum_{n=1}^{N}\left\{x^{(n)} \log \theta\right. \\
& \left.\quad+\left(1-x^{(n)}\right) \log (1-\theta)\right\}+(\alpha-1) \log \theta+(\beta-1) \log (1-\theta) \\
= & \underset{\theta}{\operatorname{argmax}}(S+\alpha-1) \log \theta+(N-S+\beta-1) \log (1-\theta),
\end{aligned}
$$

where $S=\sum_{n=1}^{N} x^{(n)}$. Taking the derivative and setting it to zero, we have

$$
\frac{S+\alpha-1}{\theta}=\frac{N-S+\beta-1}{1-\theta}
$$

Rearranging the terms we obtain the final estimate:

$$
\widehat{\theta}_{\mathrm{MAP}}=\frac{S+\alpha-1}{N+\beta+\alpha-2}
$$

There are a number of intuitions that we can draw from this beta prior, but most importantly, we have obtained a very simple solution. That is because the posterior distribution remains in the same form as the prior, after multiplying by the prior. Specifically, if we use the beta prior, the posterior distribution is

$$
\begin{aligned}
f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x}) & \propto f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta) f_{\Theta}(\theta) \\
& =\left[\prod_{n=1}^{N} \theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}\right] \cdot \frac{1}{C} \theta^{\alpha-1}(1-\theta)^{\beta-1} \\
& =\theta^{S+\alpha-1}(1-\theta)^{N-S+\beta-1} .
\end{aligned}
$$

This is still in the form of $\theta^{\star-1}(1-\theta)^{-1}$, which is the same as the prior. When this happens, we call the prior a conjugate prior. In this example, the beta prior is a conjugate before the Bernoulli likelihood.

\section{What is a conjugate prior?}

- It is a prior such that when multiplied by the likelihood to form the posterior, the posterior $f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x})$ takes the same form as the prior $f_{\Theta}(\theta)$.

- Every likelihood has its conjugate prior.

- Conjugate priors are not necessarily good priors. They are just computationally friendly. Some of them have good physical interpretations.

We can make a few interpretations of the beta prior, in the context of Bernoulli likelihood. First, the beta distribution takes the form

$$
f_{\Theta}(\theta)=\frac{1}{B(\alpha, \beta)} \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

with $B(\alpha, \beta)$ is the beta function 5 The shape of the beta distribution is shown in Figure 8.20. For different choices of $\alpha$ and $\beta$, the distribution has a peak located towards either side of the interval $[0,1]$. For example, if $\alpha$ is large but $\beta$ is small, the distribution $f_{\Theta}(\theta)$ leans towards 1 (the yellow curve).

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-531.jpg?height=601&width=840&top_left_y=1057&top_left_x=486)

Figure 8.20: Beta distribution $f_{\Theta}(\theta)$ for various choices of $\alpha$ and $\beta$. When $(\alpha, \beta)=(2,8)$, the beta distribution favors small $\theta$. When $(\alpha, \beta)=(8,2)$, the beta distribution favors large $\theta$. By swinging between the $(\alpha, \beta)$ pairs, we obtain a prior that has a preference over $\theta$.

As a user, you have the freedom to pick $f_{\Theta}(\theta)$. Even if you are restricted to the beta distribution, you still have plenty of degrees of freedom in choosing $\alpha$ and $\beta$ so that your choice matches your belief. For example, if you know ahead of time that the Bernoulli experiment is biased towards 1 (e.g., the coin is more likely to come up heads), you can choose a large $\alpha$ and a small $\beta$. By contrast, if you believe that the coin is fair, you choose $\alpha=\beta$. The parameters $\alpha$ and $\beta$ are known as the hyperparameters of the prior distribution. Hyperparameters are parameters for $f_{\Theta}(\theta)$.

${ }^{5}$ The beta function is defined as $B(\alpha, \beta)=\frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha+\beta)}$, where $\Gamma$ is the gamma function. For integer $n$, $\Gamma(n)=(n-1) !$

\section{CHAPTER 8. ESTIMATION}

Example 8.20. (Prior for Gaussian mean) Consider a Gaussian likelihood for a fixed variance $\sigma^{2}$ and unknown mean $\theta$ :

$$
f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta)=\left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)^{N} \exp \left\{-\sum_{n=1}^{N} \frac{\left(x^{(n)}-\theta\right)^{2}}{2 \sigma^{2}}\right\}
$$

Show that the conjugate prior is given by

$$
f_{\Theta}(\theta)=\frac{1}{\sqrt{2 \pi \sigma_{0}^{2}}} \exp \left\{-\frac{\left(\theta-\mu_{0}\right)^{2}}{2 \sigma_{0}^{2}}\right\}
$$

Solution. We have shown this result previously. By some (tedious) completing squares, we show that

$$
f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x})=\frac{1}{\sqrt{2 \pi \sigma_{N}^{2}}} \exp \left\{-\frac{\left(\theta-\mu_{N}\right)^{2}}{2 \sigma_{N}^{2}}\right\}
$$

where

$$
\mu_{N}=\frac{\sigma^{2}}{N \sigma_{0}^{2}+\sigma^{2}} \mu_{0}+\frac{N \sigma_{0}^{2}}{N \sigma_{0}^{2}+\sigma^{2}} \widehat{\theta} \quad \text { and } \quad \sigma_{N}^{2}=\frac{\sigma^{2} \sigma_{0}^{2}}{\sigma^{2}+N \sigma_{0}^{2}}
$$

Since $f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x})$ is in the same form as $f_{\Theta}(\theta)$, we know that $f_{\Theta}(\theta)$ is a conjugate prior.

Example 8.21. (Prior for Gaussian variance) Consider a Gaussian likelihood for a mean $\mu$ and unknown variance $\sigma^{2}$ :

$$
f_{\mathbf{X} \mid \sigma}(\mathbf{x} \mid \sigma)=\left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)^{N} \exp \left\{-\sum_{n=1}^{N} \frac{\left(x^{(n)}-\mu\right)^{2}}{2 \sigma^{2}}\right\}
$$

Find the conjugate prior.

Solution. We first define the precision $\theta=\frac{1}{\sigma^{2}}$. The likelihood is

$$
\begin{aligned}
f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta) & =\left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)^{N} \exp \left\{-\sum_{n=1}^{N} \frac{\left(x^{(n)}-\mu\right)^{2}}{2 \sigma^{2}}\right\} \\
& =\frac{1}{(2 \pi)^{N / 2}} \theta^{N / 2} \exp \left\{-\frac{\theta}{2} \sum_{n=1}^{N}\left(x^{(n)}-\mu\right)^{2}\right\} .
\end{aligned}
$$

We propose to choose the prior $f_{\Theta}(\theta)$ as

$$
f_{\Theta}(\theta)=\frac{1}{\Gamma(a)} b^{a} \theta^{a-1} \exp \{-b \theta\}
$$

for some $a$ and $b$. This $f_{\Theta}(\theta)$ is called the Gamma distribution $\operatorname{Gamma}(\theta \mid a, b)$. We can show that $\mathbb{E}[\Theta]=\frac{a}{b}$ and $\operatorname{Var}[\Theta]=\frac{a}{b^{2}}$. With some (tedious) completing squares, we

\subsection{MAXIMUM A POSTERIORI ESTIMATION}

show that the posterior is

$$
f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x}) \propto \theta^{\left(a_{0}+N / 2\right)-1} \exp \left\{-\left(b_{0}+\frac{1}{2} \sum_{n=1}^{N}\left(x^{(n)}-\mu\right)^{2}\right) \theta\right\},
$$

which is in the same form as the prior. So we know that our proposed $f_{\Theta}(\theta)$ is a conjugate prior.

The story of conjugate priors is endless because every likelihood has its conjugate prior. Table 8.1 summarizes a few commonly used conjugate priors, their likelihoods, and their posteriors. The list can be expanded further to distributions with multiple parameters. For example, if a Gaussian has both unknown mean and variance, then there exists a conjugate prior consisting of a Gaussian multiplied by a Gamma. Conjugate priors also apply to multidimensional distributions. For example, the prior for the mean vector of a high-dimensional Gaussian is another high-dimensional Gaussian. The prior for the covariance matrix of a high-dimensional Gaussian is the Wishart prior. The prior for both the mean vector and the covariance matrix is the normal Wishart.

\section{Table of Conjugate Priors}

\begin{tabular}{lll}
Likelihood & Conjugate Prior & Posterior \\
$f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta)$ & $f_{\Theta}(\theta)$ & $f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x})$ \\
\hline \hline $\operatorname{Bernoulli}(\theta)$ & $\operatorname{Beta}(\alpha, \beta)$ & $\operatorname{Beta}(\alpha+S, \beta+N-S)$ \\
$\operatorname{Poisson}(\theta)$ & $\operatorname{Gamma}(\alpha, \beta)$ & $\operatorname{Gamma}\left(\alpha+S, \frac{\beta}{1+N}\right)$ \\
$\operatorname{Exponential}(\theta)$ & $\operatorname{Gamma}(\alpha, \beta)$ & $\operatorname{Gamma}\left(\alpha+N, \frac{\beta}{1+\beta S}\right)$ \\
$\operatorname{Gaussian}\left(\theta, \sigma^{2}\right)$ & $\operatorname{Gaussian}\left(\mu_{0}, \sigma_{0}^{2}\right)$ & $\operatorname{Gaussian}\left(\frac{\mu_{0} / \sigma_{0}^{2}+S / \sigma^{2}}{1 / \sigma_{0}^{2}+N / \sigma^{2}}\right)$ \\
$\operatorname{Gaussian}\left(\mu, \theta^{2}\right)$ & $\operatorname{Inv.} \operatorname{Gamma}(\alpha, \beta)$ & $\operatorname{Gamma}\left(\alpha+\frac{N}{2}, \beta+\frac{1}{2} \sum_{n=1}^{N}\left(x^{(n)}-\mu\right)^{2}\right)$
\end{tabular}

Table 8.1: Commonly used conjugate priors.

\subsubsection{Linking MAP with regression}

ML and regression represent the statistics and the optimization aspects of the same problem. With the parallel argument, MAP is linked to the regularized regression. The reason follows immediately from the definition of MAP:

$$
\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \underbrace{\log f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})}_{\text {data fidelity }}+\underbrace{\log f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})}_{\text {regularization }} .
$$



\section{CHAPTER 8. ESTIMATION}

To make this more explicit, we consider following linear regression problem:

$$
\underbrace{\left[\begin{array}{c}
y_{1} \\
y_{2} \\
\vdots \\
y_{N}
\end{array}\right]}_{=\boldsymbol{y}}=\underbrace{\left[\begin{array}{cccc}
\phi_{0}\left(x^{(1)}\right) & \phi_{1}\left(x^{(1)}\right) & \cdots & \phi_{d-1}\left(x^{(1)}\right) \\
\phi_{0}\left(x^{(2)}\right) & \phi_{1}\left(x^{(2)}\right) & \cdots & \phi_{d-1}\left(x^{(2)}\right) \\
\vdots & \cdots & \vdots & \vdots \\
\phi_{0}\left(x^{(n)}\right) & \phi_{1}\left(x^{(n)}\right) & \cdots & \phi_{d-1}\left(x^{(n)}\right)
\end{array}\right]}_{=\mathbf{X}} \underbrace{\left[\begin{array}{c}
\theta_{0} \\
\theta_{1} \\
\vdots \\
\theta_{d-1}
\end{array}\right]}_{=\boldsymbol{\theta}}+\underbrace{\left[\begin{array}{c}
e_{1} \\
e_{2} \\
\vdots \\
e_{N}
\end{array}\right]}_{=\boldsymbol{e}} .
$$

If we assume that $\boldsymbol{e} \sim \operatorname{Gaussian}\left(0, \sigma^{2} \boldsymbol{I}\right)$, the likelihood is defined as

$$
f_{\boldsymbol{Y} \mid \boldsymbol{\Theta}}(\boldsymbol{y} \mid \boldsymbol{\theta})=\frac{1}{\sqrt{\left(2 \pi \sigma^{2}\right)^{N}}} \exp \left\{-\frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}\right\}
$$

In the ML setting, the ML estimate is the maximizer of the likelihood:

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\mathrm{ML}} & =\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \log f_{\boldsymbol{Y} \mid \boldsymbol{\Theta}}(\boldsymbol{y} \mid \boldsymbol{\theta}) \\
& =\underset{\boldsymbol{\theta}}{\operatorname{argmax}}-\frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2} .
\end{aligned}
$$

For MAP, we add a prior term so that the optimization becomes

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}} & =\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \log f_{\boldsymbol{Y} \mid \boldsymbol{\Theta}}(\boldsymbol{y} \mid \boldsymbol{\theta})+\log f_{\boldsymbol{\Theta}}(\boldsymbol{\theta}) \\
& =\underset{\boldsymbol{\theta}}{\operatorname{argmin}} \frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}-\log f_{\boldsymbol{\Theta}}(\boldsymbol{\theta}) .
\end{aligned}
$$

Therefore, the regularization of the regression is exactly $-\log f_{\Theta}(\boldsymbol{\theta})$. We can perform reverse engineering to find out the corresponding prior for our favorite choices of the regularization.

Ridge regression. Suppose that

$$
f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})=\exp \left\{-\frac{\|\boldsymbol{\theta}\|^{2}}{2 \sigma_{0}^{2}}\right\}
$$

Taking the negative log on both sides yields

$$
-\log f_{\Theta}(\boldsymbol{\theta})=\frac{\|\boldsymbol{\theta}\|^{2}}{2 \sigma_{0}^{2}}
$$

Putting this into the MAP estimate,

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}} & =\underset{\boldsymbol{\theta}}{\operatorname{argmin}} \frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}+\frac{1}{2 \sigma_{0}^{2}}\|\boldsymbol{\theta}\|^{2} \\
& =\underset{\boldsymbol{\theta}}{\operatorname{argmin}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}+\underbrace{\frac{\sigma^{2}}{\sigma_{0}^{2}}}_{=\lambda}\|\boldsymbol{\theta}\|^{2},
\end{aligned}
$$

where $\lambda$ is the corresponding ridge regularization parameter. Therefore, the ridge regression is equivalent to a MAP estimation using a Gaussian prior. How is MAP related to ridge regression?

- In MAP, define the prior as a Gaussian:

$$
f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})=\exp \left\{-\frac{\|\boldsymbol{\theta}\|^{2}}{2 \sigma_{0}^{2}}\right\}
$$

- The prior says that the solution $\boldsymbol{\theta}$ is naturally distributed according to a Gaussian with mean zero and variance $\sigma_{0}^{2}$.

LASSO regression. Suppose that

$$
f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})=\exp \left\{-\frac{\|\boldsymbol{\theta}\|_{1}}{\alpha}\right\} .
$$

Taking the negative log on both sides yields

$$
-\log f_{\Theta}(\boldsymbol{\theta})=\frac{\|\boldsymbol{\theta}\|_{1}}{\alpha}
$$

Putting this into the MAP estimate we can show that

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}} & =\underset{\boldsymbol{\theta}}{\operatorname{argmin}} \frac{1}{2 \sigma^{2}}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}+\frac{1}{\alpha}\|\boldsymbol{\theta}\|_{1} \\
& =\underset{\boldsymbol{\theta}}{\operatorname{argmin}} \frac{1}{2}\|\boldsymbol{y}-\mathbf{X} \boldsymbol{\theta}\|^{2}+\underbrace{\frac{\sigma^{2}}{\alpha}}_{=\lambda}\|\boldsymbol{\theta}\|_{1} .
\end{aligned}
$$

To summarize:

How is MAP related to LASSO regression?

- LASSO is a MAP using the prior

$$
f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})=\exp \left\{-\frac{\|\boldsymbol{\theta}\|_{1}}{\alpha}\right\} .
$$

At this point, you may be wondering what MAP buys us when regularized regression can already do the job. The answer is about the interpretation. While regularized regression can always return us a result, that is just a result. However, if you know that the parameter $\boldsymbol{\theta}$ is distributed according to some distributions $f_{\Theta}(\boldsymbol{\theta})$, MAP offers a statistical perspective of the solution in the sense that it returns the peak of the posterior $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$. For example, if we know that the data is generated from a linear model with Gaussian noise, and if we know that the true regression coefficients are drawn from a Gaussian, then the ridge regression is guaranteed to be optimal in the posterior sense. Similarly, if we know that there are outliers and have some ideas about the outlier statistics, perhaps the LASSO regression is a better choice.

It is also important to note the different optimalities offered by MAP versus ML versus regression. The optimality offered by regression is the training loss, which can always give us a result even if the underlying statistics do not match the optimization formulation,

\section{CHAPTER 8. ESTIMATION}

e.g., there are outliers, and you use unregularized least-squares minimization. You can get a result, but the outliers will heavily influence your solution. On the other hand, if you know the data statistics and choose to follow the ML, then the ML solution is optimal in the sense of optimizing the likelihood $f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})$. If you further know the prior statistics, the MAP solution will be optimal, but this time it is optimal w.r.t. to the posterior $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$. Since each of these is optimizing for a different goal, they are only good for their chosen objectives. For example, $\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}$ can be a biased estimate if our goal is to maximize the likelihood. The $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}$ is optimal for the likelihood but can be a bad choice for the posterior. Both $\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}$ and $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}$ can possibly achieve a reasonable mean-squared error, but their results may not make sense (e.g., if $\boldsymbol{\theta}$ is an image then $\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}$ may over-smooth the image whereas $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}$ amplifies noise). So it's incorrect to think that $\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}$ is superior to $\widehat{\boldsymbol{\theta}}_{\mathrm{ML}}$ because it is more general.

Here are some rules of thumb for MAP, ML, and regression:

When should I use regression, ML and MAP?

- Regression: If you are lazy and you know nothing about the statistics, do the regression with whatever regularization you prefer. It will give you a result. See if it makes sense with your data.

- MAP: If you know the statistics of the data, and if you have some preference for the prior distribution, go with MAP. It will offer you the optimal solution w.r.t. finding the peak of the posterior.

- ML: If you are interested in some simple-form solution, and you want those nice properties such as consistency and unbiasedness, then go with ML. It usually possesses the "friendly" properties so that you can derive the performance limit.

\subsection{Minimum Mean-Square Estimation}

First-time readers are often tempted to think that the maximum-likelihood estimation or the maximum a posteriori estimation are the best methods to estimate parameters. In some sense, this is true because both estimation procedures offer some form of optimal explanation for the observed variables. However, as we said above, being optimal with respect to the likelihood or the posterior only means optimal under the respective criteria. An ML estimate is not necessarily optimal for the posterior, whereas a MAP estimate is not necessarily optimal for the likelihood. Therefore, as we proceed to the third commonly used estimation strategy, we need to remind ourselves of the specific type of optimality we seek.

\subsubsection{Positioning the minimum mean-square estimation}

Mean-square error estimation, as it is termed, uses the mean-square error as the optimality criterion. The corresponding estimation process is known as the minimum mean-square estimation (MMSE). MMSE is a Bayesian approach, meaning that it uses the prior $f_{\Theta}(\boldsymbol{\theta})$ as well as the likelihood $f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})$. As we will show shortly, the MMSE estimate of a set

\subsection{MINIMUM MEAN-SQUARE ESTIMATION}

of i.i.d. observation $\mathbf{X}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$ is

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\mathrm{MMSE}}(\mathbf{x}) & \stackrel{(a)}{=} \mathbb{E}_{\boldsymbol{\Theta} \mid \mathbf{X}}[\boldsymbol{\Theta} \mid \mathbf{X}=\mathbf{x}] \quad(a): \text { We will discuss this. } \\
& =\int \boldsymbol{\theta} \cdot f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x}) d \boldsymbol{\theta} .
\end{aligned}
$$

You may find this equation very surprising, because it says that the MMSE estimate is the mean of the posterior distribution $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$. Let's compare this result with the ML estimate and the MAP estimate:

$$
\begin{aligned}
& \widehat{\boldsymbol{\theta}}_{\mathrm{ML}}=\text { peak of } f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta}), \\
& \widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}=\text { peak of } f_{\boldsymbol{\Theta} \mathbf{X} \mid}(\boldsymbol{\theta} \mid \mathbf{x}), \\
& \widehat{\boldsymbol{\theta}}_{\mathrm{MMSE}}=\text { average of } f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x}) .
\end{aligned}
$$

Therefore, an MMSE estimate is not by any means universally superior or inferior to a MAP estimate or an ML estimate. It is just a different estimate with a different goal.

So how exactly are these estimates different? Figure 8.21 illustrates a typical situation of asymmetric distribution. Here, we plot both the likelihood function $f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})$ and the posterior function $f_{\Theta \boldsymbol{\Theta} \mid}(\boldsymbol{\theta} \mid \mathbf{x})$.

![](https://cdn.mathpix.com/cropped/2023_03_13_39570ad67b23e5b59046g-537.jpg?height=466&width=574&top_left_y=1105&top_left_x=621)

Figure 8.21: A typical example of an ML estimate, a MAP estimate and an MMSE estimate.

As shown in the figure, the ML estimate is the peak of the likelihood, whereas the MAP estimate is the peak of the posterior. The third estimate is the MMSE estimate, which is the average of the posterior distribution. It is easy to see that if the posterior distribution is symmetric and has a single peak, the peak is always the mean. Therefore, for single-peak symmetric distributions, MMSE and MAP estimates are identical.

What is so special about the MMSE estimate?

- MMSE is a Bayesian estimation, so it requires a prior.

- An MMSE estimate is the mean of the posterior distribution.

- MMSE estimate = MAP estimate if the posterior distribution is symmetric and has a single peak.

\section{CHAPTER 8. ESTIMATION}

\subsubsection{Mean squared error}

The MMSE is based on minimizing the mean squared error (MSE). In this subsection we discuss the mean squared error in the Bayesian setting. In the deterministic setting, given an estimate $\widehat{\theta}$ and a ground truth $\theta$, the MSE is defined as

$$
\operatorname{MSE}(\underbrace{\theta}_{\text {ground truth }}, \underbrace{\hat{\theta}}_{\text {estimate }})=(\theta-\widehat{\theta})^{2} .
$$

In any estimation problem, the estimate $\widehat{\theta}$ is always a function of the observed variables. Thus, we have

$$
\widehat{\theta}(\mathbf{X})=g(\mathbf{X}), \quad \text { where } \quad \mathbf{X}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}
$$

for some function $g(\cdot)$. Substituting this into the definition of MSE, and recognizing that $\mathbf{X}$ is drawn from a distribution $f_{\mathbf{X}}(\mathbf{x})$, we take the expectation to define the MSE as

$$
\begin{aligned}
\operatorname{MSE}(\theta, \widehat{\theta}) & =(\theta-\widehat{\theta})^{2} \\
& \Downarrow \text { replace } \widehat{\theta} \text { by } g(\mathbf{X}) \\
\operatorname{MSE}(\theta, \widehat{\theta}) & =(\theta-g(\mathbf{X}))^{2} \\
& \Downarrow \text { take expectation over } \mathbf{X} \\
\operatorname{MSE}(\theta, \widehat{\theta}) & =\mathbb{E}_{\mathbf{X}}\left[(\theta-g(\mathbf{X}))^{2}\right] .
\end{aligned}
$$

Thus we have arrived at the definition of MSE. We call this the frequentist version, because the parameter $\theta$ is deterministic.

Definition 8.8 (Mean squared error, frequentist). The mean squared error of an estimate $g(\mathbf{X})$ w.r.t. the true parameter $\theta$ is

$$
M S E_{f r e q}(\theta, g(\cdot))=\mathbb{E}_{\mathbf{X}}\left[(\theta-g(\mathbf{X}))^{2}\right]
$$

If the parameter $\boldsymbol{\theta}$ is high-dimensional, so is the estimate $\boldsymbol{g}(\mathbf{X})$, and the $M S E$ is

$$
M S E_{\text {freq }}(\boldsymbol{\theta}, \boldsymbol{g}(\cdot))=\mathbb{E}_{\mathbf{X}}\left[\|\boldsymbol{\theta}-\boldsymbol{g}(\mathbf{X})\|^{2}\right]
$$

Note that in the above definition the MSE is measured between the true parameter $\theta$ and the estimator $g(\cdot)$. We use the function $g(\cdot)$ here because we have taken the expectation of all the possible inputs $\mathbf{X}$. So we are not comparing $\theta$ with a value $g(\mathbf{X})$ but with the function $g(\cdot)$.

If we take a Bayesian approach such as the MAP, then $\theta$ itself is a random variable $\Theta$. To compute the MSE, we then need to take the average across all the possible choices of ground truth $\Theta$. This leads to

$$
\begin{aligned}
\operatorname{MSE}(\theta, \widehat{\theta}) & =\mathbb{E}_{\mathbf{X}}\left[(\theta-g(\mathbf{X}))^{2}\right] \\
& \Downarrow \text { replace } \theta \text { by } \Theta \\
\operatorname{MSE}(\theta, \widehat{\theta}) & =\mathbb{E}_{\mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2}\right] \\
& \Downarrow \text { take expectation over } \Theta \\
\operatorname{MSE}(\theta, \widehat{\theta}) & =\mathbb{E}_{\mathbf{X}, \Theta}\left[(\Theta-g(\mathbf{X}))^{2}\right] .
\end{aligned}
$$

Therefore, we have arrived at our definition of the MSE, in the Bayesian setting. Definition 8.9 (Mean squared error, Bayesian). The mean squared error of an estimate $g(\mathbf{X})$ w.r.t. the true parameter $\Theta$ is

$$
M S E_{\text {Bayes }}(\Theta, g(\cdot))=\mathbb{E}_{\Theta, \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2}\right]
$$

If the parameter $\boldsymbol{\Theta}$ is high-dimensional, so is the estimate $\boldsymbol{g}(\mathbf{X})$, and the MSE is

$$
M S E_{\text {Bayes }}(\boldsymbol{\Theta}, \boldsymbol{g}(\cdot))=\mathbb{E}_{\boldsymbol{\Theta}, \mathbf{X}}\left[\|\boldsymbol{\Theta}-\boldsymbol{g}(\mathbf{X})\|^{2}\right]
$$

The difference between the Bayesian MSE and the frequentist MSE is the expectation over $\Theta$. Practically speaking, the frequentist MSE is more of an evaluation metric than an objective function for solving an inverse problem. The reason is that in an inverse problem, we never have access to the true parameter $\theta$. (If we knew $\theta$, there would be no problem to solve.) Bayesian MSE is more meaningful. It says that we do not know the true parameter $\theta$, but we know its statistics. We are trying to find the best $g(\cdot)$ that minimizes the error. Our solution will depend on the statistics of $\Theta$ but not on the unknown true parameter $\theta$.

When we say minimum mean squared error estimation, we typically refer to the Bayesian MMSE. In this case, the problem we solve is

$$
g(\cdot)=\underset{g(\cdot)}{\operatorname{argmin}} \mathbb{E}_{\Theta, \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2}\right]
$$

As you can see from Definition 8.9, the goal of the Bayesian MMSE is to find a function $g: \mathbb{R}^{N} \rightarrow \mathbb{R}$ such that the joint expectation $\mathbb{E}_{\Theta, \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2}\right]$ is minimized. In the case where $\Theta$ is a vector, the problem becomes

$$
\boldsymbol{g}(\cdot)=\underset{\boldsymbol{g}(\cdot)}{\operatorname{argmin}} \mathbb{E}_{\boldsymbol{\Theta}, \mathbf{X}}\left[\|\boldsymbol{\Theta}-\boldsymbol{g}(\mathbf{X})\|^{2}\right]
$$

where $\boldsymbol{g}(\cdot): \mathbb{R}^{N \times d} \rightarrow \mathbb{R}^{d}$ if $\boldsymbol{\Theta}$ is a $d$-dimensional vector. The function $\boldsymbol{g}$ will take a sequence of $N$ observed numbers and estimate the parameter $\boldsymbol{\Theta}$.

What is the Bayesian MMSE estimate?

The Bayesian MMSE estimate is obtained by minimizing the MSE:

$$
g(\cdot)=\underset{g(\cdot)}{\operatorname{argmin}} \mathbb{E}_{\Theta, \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2}\right]
$$

\subsubsection{MMSE estimate $=$ conditional expectation}

Theorem 8.2. The Bayesian MMSE estimate is

$$
\begin{aligned}
\widehat{\theta}_{M M S E} & =\underset{g(\cdot)}{\operatorname{argmin}} \mathbb{E}_{\Theta, \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2}\right] \\
& =\mathbb{E}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}=\mathbf{x}] .
\end{aligned}
$$



\section{CHAPTER 8. ESTIMATION}

Proof. First of all, we decompose the joint expectation:

$$
\mathbb{E}_{\Theta, \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2}\right]=\int \mathbb{E}_{\Theta \mid \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2} \mid \mathbf{X}=\mathbf{x}\right] f_{\mathbf{X}}(\mathbf{x}) d \mathbf{x}
$$

Since $f_{\mathbf{X}}(\mathbf{x}) \geq 0$ for all $\mathbf{x}$, and $\mathbb{E}_{\Theta \mid \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2} \mid \mathbf{X}=\mathbf{x}\right] \geq 0$ because it is a square, it follows that the integral is minimized when $\mathbb{E}_{\Theta \mid \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2} \mid \mathbf{X}=\mathbf{x}\right]$ is minimized.

The conditional expectation can be evaluated as

$$
\begin{aligned}
& \mathbb{E}_{\Theta \mid \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2} \mid \mathbf{X}=\mathbf{x}\right] \\
& =\mathbb{E}_{\Theta \mid \mathbf{X}}\left[\Theta^{2}-2 \Theta g(\mathbf{X})+g(\mathbf{X})^{2} \mid \mathbf{X}=\mathbf{x}\right] \\
& =\underbrace{\mathbb{E}_{\Theta \mid \mathbf{X}}\left[\Theta^{2} \mid \mathbf{X}=\mathbf{x}\right]}_{\stackrel{\text { def }}{=} V(\mathbf{x})}-\underbrace{2 \mathbb{E}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}=\mathbf{x}]}_{\stackrel{\text { def }}{=} u(\mathbf{x})} g(\mathbf{x})+g(\mathbf{x})^{2} \\
& =V(\mathbf{x})-2 u(\mathbf{x}) g(\mathbf{x})+g(\mathbf{x})^{2}+u(\mathbf{x})^{2}-u(\mathbf{x})^{2} \\
& =V(\mathbf{x})-u(\mathbf{x})^{2}+(u(\mathbf{x})-g(\mathbf{x}))^{2} \\
& \geq V(\mathbf{x})-u(\mathbf{x})^{2}, \quad \forall g(\cdot),
\end{aligned}
$$

where the last inequality holds because no matter what $g(\cdot)$ we choose, the square term $(u(\mathbf{x})-g(\mathbf{x}))^{2}$ is non-negative. Therefore, $\mathbb{E}_{\Theta \mid \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2} \mid \mathbf{X}=\mathbf{x}\right]$ is lower-bounded by $V(\mathbf{x})-u(\mathbf{x})^{2}$, which is a bound that is independent of $g(\cdot)$. If we can find a $g(\cdot)$ such that this lower bound can be met, the corresponding $g(\cdot)$ is the minimizer.

To this end we only need to make $\mathbb{E}_{\Theta \mid \mathbf{X}}\left[(\Theta-g(\mathbf{X}))^{2} \mid \mathbf{X}=\mathbf{x}\right]$ equal $V(\mathbf{x})-u(\mathbf{x})^{2}$, but this is easy: the equality holds if and only if $(u(\mathbf{x})-g(\mathbf{x}))^{2}=0$. In other words, if we choose $g(\cdot)$ such that $g(\mathbf{x})=u(\mathbf{x})$, the corresponding $g(\cdot)$ is the minimizer. This $g(\cdot)$, by substituting the definition of $u(\mathbf{x})$, is

$$
g(\mathbf{x})=\mathbb{E}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}=\mathbf{x}]
$$

This completes the proof.

What is the MMSE estimate?

The MMSE estimate is

$$
\widehat{\theta}_{\operatorname{MMSE}}(\mathbf{x})=\mathbb{E}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}=\mathbf{x}]
$$

We emphasize that $\widehat{\theta}_{\operatorname{MMSE}}(\mathbf{x})$ is a function of $\mathbf{x}$, because for a different set of observations $\mathbf{x}$ we will have a different estimated value. Since $\mathbf{x}$ is a random realization of the random vector $\mathbf{X}$, we can also define the MMSE estimator as

$$
\widehat{\Theta}_{\operatorname{MMSE}}(\mathbf{X})=\mathbb{E}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}]
$$

In this notation, we emphasize that the estimator $\widehat{\Theta}_{\text {MMSE }}$ returns a random parameter. The input to the estimator is the random vector $\mathbf{X}$. Because we are not looking at a particular realization $\mathbf{X}=\mathbf{x}$ but the general $\mathbf{X}, \widehat{\Theta}_{\text {MMSE }}$ is a function of $\mathbf{X}$ and not $\mathbf{x}$. Conditional expectation of what?

An MMSE estimator is the conditional expectation of $\Theta$ given $\mathbf{X}=\mathbf{x}$ :

$$
\mathbb{E}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}=\mathbf{x}]=\int \theta f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x}) d \theta
$$

This is the expectation using the posterior distribution $f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x})$. It should be compared to the peak of the posterior, which returns us the MAP estimate. The posterior distribution is constructed through Bayes' theorem:

$$
f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x})=\frac{f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta) f_{\Theta}(\theta)}{f_{\mathbf{X}}(\mathbf{x})}
$$

Therefore, to evaluate the expectation of the condition distribution, we need to include the normalization constant $f_{\mathbf{X}}(\mathbf{x})$, which was omitted in MAP.

The discussion about the mean squared error and the vector estimates can be skipped if this is your first time reading the book.

What is the mean squared error when using the MMSE estimator?

- The mean squared error conditioned on the observation is

$$
\begin{aligned}
\operatorname{MSE}\left(\Theta, \widehat{\Theta}_{\operatorname{MMSE}}(\mathbf{X})\right) & \stackrel{\text { def }}{=} \mathbb{E}_{\Theta \mid \mathbf{X}}\left[\left(\Theta-\widehat{\Theta}_{\operatorname{MMSE}}(\mathbf{X})\right)^{2} \mid \mathbf{X}\right] \\
& =\operatorname{Var}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}]
\end{aligned}
$$

which is the conditional variance.

- The overall mean squared error, unconditioned, is

$$
\begin{aligned}
\operatorname{MSE}\left(\Theta, \widehat{\Theta}_{\operatorname{MMSE}}(\cdot)\right) & =\mathbb{E}_{\mathbf{X}}\left[\operatorname{Var}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}]\right] \\
& =\operatorname{Var}_{\Theta}[\Theta] .
\end{aligned}
$$

Proof. Let us prove these two statements. The resulting MSE is obtained by substituting $\widehat{\Theta}_{\operatorname{MMSE}}(\mathbf{x})=\mathbb{E}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}]$ into the $\operatorname{MSE}\left(\Theta, \widehat{\Theta}_{\operatorname{MMSE}}(\mathbf{X})\right)$. To this end, we have that

$$
\begin{aligned}
\mathbb{E}_{\Theta \mid \mathbf{X}}\left[\left(\Theta-\widehat{\Theta}_{\operatorname{MMSE}}(\mathbf{X})\right)^{2} \mid \mathbf{X}\right]=V(\mathbf{X}) & -u(\mathbf{X})^{2} \\
& +\underbrace{\left(u(\mathbf{X})-\widehat{\Theta}_{\mathrm{MMSE}}(\mathbf{X})\right)^{2}}_{=0, \text { because } \widehat{\Theta}_{\mathrm{MMSE}}(\mathbf{X})=\mathbb{E}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}]=u(\mathbf{X})} .
\end{aligned}
$$

The variables $V$ and $u$ are defined as

$$
\begin{aligned}
V(\mathbf{X}) & =\mathbb{E}_{\Theta \mid \mathbf{X}}\left[\Theta^{2} \mid \mathbf{X}\right]=2 \text { nd moment of } \Theta \text { using } f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x}), \\
u(\mathbf{X}) & =\mathbb{E}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}]=1 \text { st moment of } \Theta \text { using } f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x}) .
\end{aligned}
$$



\section{CHAPTER 8. ESTIMATION}

Since $\operatorname{Var}[Z]=\mathbb{E}\left[Z^{2}\right]-\mathbb{E}[Z]^{2}$ for any random variable $Z$, it follows that

$$
\begin{aligned}
\mathbb{E}_{\Theta \mid \mathbf{X}}\left[\left(\Theta-\widehat{\Theta}_{\operatorname{MMSE}}(\mathbf{X})\right)^{2} \mid \mathbf{X}\right] & =V(\mathbf{X})-u(\mathbf{X})^{2} \\
& =\mathbb{E}_{\Theta \mid \mathbf{X}}\left[\Theta^{2} \mid \mathbf{X}\right]-\left(\mathbb{E}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}]\right)^{2} \\
& =\text { variance of } \Theta \text { using } f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x}) \\
& \stackrel{\text { def }}{=} \operatorname{Var}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}] .
\end{aligned}
$$

Substituting this conditional variance into the MSE definition,

$$
\begin{aligned}
\operatorname{MSE}\left(\Theta, \widehat{\Theta}_{\operatorname{MMSE}}(\cdot)\right) & =\int \mathbb{E}_{\Theta \mid \mathbf{X}}\left[\left(\Theta-\widehat{\Theta}_{\operatorname{MMSE}}(\mathbf{X})\right)^{2} \mid \mathbf{X}=\mathbf{x}\right] f_{\mathbf{X}}(\mathbf{x}) d \mathbf{x} \\
& =\int \operatorname{Var}_{\Theta \mid \mathbf{X}}[\Theta \mid \mathbf{X}=\mathbf{x}] f_{\mathbf{X}}(\mathbf{x}) d \mathbf{x} \\
& =\operatorname{Var}_{\Theta}[\Theta] .
\end{aligned}
$$

What happens if the parameter is a vector?

- The MMSE estimate is $\widehat{\boldsymbol{\theta}}_{\mathrm{MMSE}}(\mathbf{x})=\mathbb{E}_{\boldsymbol{\Theta} \mid \mathbf{X}}[\boldsymbol{\Theta} \mid \mathbf{X}=\mathbf{x}]$.

- The MSE is

$$
\operatorname{MSE}\left(\boldsymbol{\Theta}, \widehat{\boldsymbol{\Theta}}_{\mathrm{MMSE}}(\cdot)\right)=\operatorname{Tr}\left\{\mathbb{E}_{\mathbf{X}}\{\operatorname{Cov}(\boldsymbol{\Theta} \mid \mathbf{X})\}\right\}
$$

Proof. The first statement, that the MMSE estimate is

$$
\widehat{\boldsymbol{\theta}}_{\operatorname{MMSE}}(\mathbf{x})=\mathbb{E}_{\boldsymbol{\Theta} \mid \mathbf{X}}[\boldsymbol{\Theta} \mid \mathbf{X}=\mathbf{x}]
$$

is easy to understand since it just follows from the scalar case. The estimator is $\widehat{\boldsymbol{\Theta}}_{\mathrm{MMSE}}(\mathbf{X})=$ $\mathbb{E}_{\boldsymbol{\Theta} \mid \mathbf{X}}[\boldsymbol{\Theta} \mid \mathbf{X}]$. The corresponding MSE is

$$
\begin{aligned}
\operatorname{MSE}\left(\boldsymbol{\Theta}, \widehat{\boldsymbol{\Theta}}_{\mathrm{MMSE}}(\cdot)\right) & =\mathbb{E}_{\boldsymbol{\Theta}, \mathbf{X}}\left[\left\|\boldsymbol{\Theta}-\widehat{\boldsymbol{\Theta}}_{\mathrm{MMSE}}(\mathbf{X})\right\|^{2}\right] \\
& =\mathbb{E}_{\mathbf{X}}\left\{\mathbb{E}_{\boldsymbol{\Theta} \mid \mathbf{X}}\left[\left\|\boldsymbol{\Theta}-\widehat{\boldsymbol{\Theta}}_{\operatorname{MMSE}}(\mathbf{X})\right\|^{2} \mid \mathbf{X}\right]\right\},
\end{aligned}
$$

where we have used the law of total expectation to decompose the joint expectation. Using the matrix identity below, we have that

$$
\begin{aligned}
& \mathbb{E}_{\mathbf{X}}\left\{\mathbb{E}_{\boldsymbol{\Theta} \mid \mathbf{X}}\left[\left\|\boldsymbol{\Theta}-\widehat{\boldsymbol{\Theta}}_{\mathrm{MMSE}}(\mathbf{X})\right\|^{2} \mid \mathbf{X}\right]\right\} \\
& =\mathbb{E}_{\mathbf{X}}\left\{\mathbb{E}_{\boldsymbol{\Theta} \mid \mathbf{X}}\left[\operatorname{Tr}\left\{\left(\boldsymbol{\Theta}-\widehat{\boldsymbol{\Theta}}_{\mathrm{MMSE}}(\mathbf{X})\right)\left(\boldsymbol{\Theta}-\widehat{\boldsymbol{\Theta}}_{\mathrm{MMSE}}(\mathbf{X})\right)^{T}\right\} \mid \mathbf{X}\right]\right\} \\
& =\operatorname{Tr}\left\{\mathbb{E}_{\mathbf{X}}\left\{\mathbb{E}_{\boldsymbol{\Theta} \mid \mathbf{X}}\left[\left(\boldsymbol{\Theta}-\widehat{\boldsymbol{\Theta}}_{\mathrm{MMSE}}(\mathbf{X})\right)\left(\boldsymbol{\Theta}-\widehat{\boldsymbol{\Theta}}_{\mathrm{MMSE}}(\mathbf{X})\right)^{T} \mid \mathbf{X}\right]\right\}\right\} .
\end{aligned}
$$



\subsection{MINIMUM MEAN-SQUARE ESTIMATION}

However, since the MMSE estimator is the condition expectation of the posterior, it follows that the inner expectation is the conditional covariance. Therefore, we arrive at the second statement:

$$
\begin{aligned}
\operatorname{MSE}\left(\boldsymbol{\Theta}, \widehat{\boldsymbol{\Theta}}_{\operatorname{MMSE}}(\cdot)\right) & =\operatorname{Tr}\left\{\mathbb{E}_{\mathbf{X}}\left\{\mathbb{E}_{\boldsymbol{\Theta} \mid \mathbf{X}}\left[\left(\boldsymbol{\Theta}-\widehat{\boldsymbol{\Theta}}_{\operatorname{MMSE}}(\mathbf{X})\right)\left(\boldsymbol{\Theta}-\widehat{\boldsymbol{\Theta}}_{\operatorname{MMSE}}(\mathbf{X})\right)^{T} \mid \mathbf{X}\right]\right\}\right\} \\
& =\operatorname{Tr}\left\{\mathbb{E}_{\mathbf{X}}\{\operatorname{Cov}(\boldsymbol{\Theta} \mid \mathbf{X})\}\right\}
\end{aligned}
$$

To prove the two statements above, we need some tools from linear algebra. The two specific matrix identities are given by the following lemma:

Lemma 8.1. The following are matrix identities:

- For any random vector $\boldsymbol{\Theta} \in \mathbb{R}^{d}$,

$$
\|\boldsymbol{\Theta}\|^{2}=\operatorname{Tr}\left(\boldsymbol{\Theta}^{T} \boldsymbol{\Theta}\right)=\operatorname{Tr}\left(\boldsymbol{\Theta} \Theta^{T}\right) .
$$

- For any random vector $\boldsymbol{\Theta} \in \mathbb{R}^{d}$,

$$
\mathbb{E}_{\boldsymbol{\Theta}}\left[\operatorname{Tr}\left(\boldsymbol{\Theta} \Theta^{T}\right)\right]=\operatorname{Tr}\left(\mathbb{E}_{\boldsymbol{\Theta}}\left[\boldsymbol{\Theta} \Theta^{T}\right]\right)
$$

The proof of these two results is straightforward. The first is due to the cyclic property of the trace operator. The second statement is true because the trace is a linear operator that sums the diagonal of a matrix.

The end of the discussion. Please join us again.

Example 8.22. Let

$$
f_{X \mid \Theta}(x \mid \theta)=\left\{\begin{array}{ll}
\theta e^{-\theta x}, & x \geq 0, \\
0, & x<0,
\end{array} \quad \text { and } \quad f_{\Theta}(\theta)= \begin{cases}\alpha e^{-\alpha \theta}, & \theta \geq 0, \\
0, & \theta<0 .\end{cases}\right.
$$

Find the ML, MAP, and MMSE estimates for a single observation $X=x$.

Solution. We first find the posterior distribution:

$$
\begin{aligned}
f_{\Theta \mid X}(\theta \mid x) & =\frac{f_{X \mid \Theta}(x \mid \theta) f_{\Theta}(\theta)}{f_{X}(x)} \\
& =\frac{\alpha \theta e^{-(\alpha+x) \theta}}{\int_{0}^{\infty} \alpha \theta e^{-(\alpha+x) \theta} d \theta} \\
& =\frac{\alpha \theta e^{-(\alpha+x) \theta}}{\frac{\alpha}{(\alpha+x)^{2}}} \\
& =(\alpha+x)^{2} \theta e^{-(\alpha+x) \theta}
\end{aligned}
$$



\section{CHAPTER 8. ESTIMATION}

The MMSE estimate is the conditional expectation of the posterior:

$$
\begin{aligned}
\widehat{\theta}_{\operatorname{MMSE}}(x) & =\mathbb{E}_{\Theta \mid X}[\Theta \mid X=x] \\
& =\int_{0}^{\infty} \theta f_{\Theta \mid X}(\theta \mid x) d \theta \\
& =\int_{0}^{\infty} \theta(\alpha+x)^{2} \theta e^{-(\alpha+x) \theta} d \theta \\
& =(\alpha+x) \underbrace{\int_{0}^{\infty} \theta^{2} \cdot(\alpha+x) e^{-(\alpha+x) \theta} d \theta}_{2 \text { nd moment of exponential distribution }} \\
& =(\alpha+x) \cdot \frac{2}{(\alpha+x)^{2}}=\frac{2}{\alpha+x} .
\end{aligned}
$$

The MAP estimate is the peak of the posterior:

$$
\begin{array}{rlr}
\widehat{\theta}_{\mathrm{MAP}}(x) & =\underset{\theta}{\operatorname{argmax}} & \log f_{X \mid \Theta}(x \mid \theta)+\log f_{\Theta}(\theta) \\
& =\underset{\theta}{\operatorname{argmax}} & -\theta x+\log \theta-\alpha \theta+\log \alpha .
\end{array}
$$

Taking the derivative and setting it to zero yields $-x+\frac{1}{\theta}-\alpha=0$. This implies that

$$
\widehat{\theta}_{\mathrm{MAP}}(x)=\frac{1}{\alpha+x}
$$

Finally, the ML estimate is

$$
\widehat{\theta}(x)=\underset{\theta}{\operatorname{argmax}} \log f_{X \mid \Theta}(x \mid \theta)=\frac{1}{x} .
$$

Practice Exercise 8.8. Following the previous example, derive the estimates for multiple observations $\mathbf{X}=\mathbf{x}$.

Solution. The posterior is

$$
\begin{aligned}
f_{\Theta \mid \mathbf{X}}(\theta \mid \mathbf{x}) & =\frac{f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \theta) f_{\Theta}(\theta)}{f_{\mathbf{X}}(\mathbf{x})} \\
& =\frac{\left(\prod_{n=1}^{N} f_{X \mid \Theta}\left(x^{(n)} \mid \theta\right)\right) f_{\Theta}(\theta)}{f_{\mathbf{X}}(\mathbf{x})} \\
& =\frac{\alpha \theta e^{-\left(\alpha+\sum_{n=1}^{N} x^{(n)}\right) \theta}}{\int_{0}^{\infty} \alpha \theta e^{-\left(\alpha+\sum_{n=1}^{N} x^{(n)}\right) \theta} d \theta} \\
& =\left(\alpha+\sum_{n=1}^{N} x^{(n)}\right)^{2} \theta e^{-\left(\alpha+\sum_{n=1}^{N} x^{(n)}\right) \theta}
\end{aligned}
$$



\subsection{MINIMUM MEAN-SQUARE ESTIMATION}

Therefore, we are only replacing $x$ by the sum $\sum_{n=1}^{N} x^{(n)}$ in the posterior. Hence, the estimates are:

$$
\begin{aligned}
\widehat{\theta}_{\mathrm{MMSE}}(x) & =\frac{2}{\alpha+\sum_{n=1}^{N} x^{(n)}}, \\
\widehat{\theta}_{\mathrm{MAP}}(x) & =\frac{1}{\alpha+\sum_{n=1}^{N} x^{(n)}}, \\
\widehat{\theta}(x) & =\frac{1}{\sum_{n=1}^{N} x^{(n)}}
\end{aligned}
$$

This example shows that as $N \rightarrow \infty$, the ML estimate $\widehat{\theta}(x) \rightarrow 0$. The reason is that the likelihood is an exponential distribution. Therefore, the peak is always at 0 . The posterior is an Erlang distribution, and therefore the peak is offset by $\alpha$ in the denominator. However, as $N \rightarrow \infty$ the posterior distribution is dominated by the likelihood, so the peak is shifted towards 0. Finally, since the Erlang distribution is asymmetric, the mean is different from the peak. Hence, the MMSE estimate is different from the MAP estimate.

\subsubsection{MMSE estimator for multidimensional Gaussian}

The multidimensional Gaussian has some very important uses in data science. Accordingly, we devote this subsection to the discussion of the MMSE estimate of a Gaussian. The main result is stated as follows.

What is the MMSE estimator for a multi-dimensional Gaussian?

Theorem 8.3. Suppose $\boldsymbol{\Theta} \in \mathbb{R}^{d}$ and $\mathbf{X} \in \mathbb{R}^{N}$ are jointly Gaussian with a joint PDF

$$
\left[\begin{array}{l}
\boldsymbol{\Theta} \\
\mathbf{X}
\end{array}\right] \sim \operatorname{Gaussian}\left(\left[\begin{array}{l}
\boldsymbol{\mu}_{\Theta} \\
\boldsymbol{\mu}_{X}
\end{array}\right],\left[\begin{array}{ll}
\boldsymbol{\Sigma}_{\Theta \Theta} & \boldsymbol{\Sigma}_{\Theta X} \\
\boldsymbol{\Sigma}_{X \Theta} & \boldsymbol{\Sigma}_{X X}
\end{array}\right]\right) .
$$

The MMSE estimator is

$$
\widehat{\boldsymbol{\Theta}}_{M M S E}(\mathbf{X})=\boldsymbol{\mu}_{\Theta}+\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1}\left(\mathbf{X}-\boldsymbol{\mu}_{X}\right)
$$

The proof of this result is not difficult but it is tedious. The flow of the argument is:

- Step 1: Show that the posterior distribution $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$ is a Gaussian.

- Step 2: To do so we need to complete the squares for matrices.

- Step 3: Once we have the $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$, the posterior mean is the MMSE estimator.

The proof below can be skipped if this is your first time reading the book.

\section{CHAPTER 8. ESTIMATION}

Proof. The posterior PDF is

$$
\begin{aligned}
& f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})=\frac{f_{\boldsymbol{\Theta}, \mathbf{X}}(\boldsymbol{\theta}, \mathbf{x})}{f_{\mathbf{X}}(\mathbf{x})} \\
& =\frac{\frac{1}{\sqrt{(2 \pi)^{d+N}|\boldsymbol{\Sigma}|}} \exp \left\{-\frac{1}{2}\left[\begin{array}{l}
\boldsymbol{\theta}-\boldsymbol{\mu}_{\Theta} \\
\mathbf{x}-\boldsymbol{\mu}_{X}
\end{array}\right]^{T}\left[\begin{array}{ll}
\boldsymbol{\Sigma}_{\Theta \Theta} & \boldsymbol{\Sigma}_{\Theta X} \\
\boldsymbol{\Sigma}_{X \Theta} & \boldsymbol{\Sigma}_{X X}
\end{array}\right]^{-1}\left[\begin{array}{l}
\boldsymbol{\theta}-\boldsymbol{\mu}_{\Theta} \\
\mathbf{x}-\boldsymbol{\mu}_{X}
\end{array}\right]\right\}}{\frac{1}{\sqrt{(2 \pi)^{N}\left|\boldsymbol{\Sigma}_{X X}\right|}} \exp \left\{-\frac{1}{2}\left[\mathbf{x}-\boldsymbol{\mu}_{X}\right]^{T} \boldsymbol{\Sigma}_{X X}^{-1}\left[\mathbf{x}-\boldsymbol{\mu}_{X}\right]\right\}} .
\end{aligned}
$$

Without loss of generality, we assume that $\boldsymbol{\mu}_{X}=\boldsymbol{\mu}_{\Theta}=0$. Then the posterior becomes

$$
\begin{aligned}
f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})= & \frac{1}{\sqrt{(2 \pi)^{d}|\boldsymbol{\Sigma}| /\left|\boldsymbol{\Sigma}_{X X}\right|}} \\
& \times \exp \underbrace{\left\{\begin{array}{ll}
-\frac{1}{2}\left[\begin{array}{l}
\boldsymbol{\theta} \\
\mathbf{x}
\end{array}\right]^{T}
\end{array}\left[\begin{array}{cc}
\boldsymbol{\Sigma}_{\Theta \Theta} & \boldsymbol{\Sigma}_{\Theta X} \\
\boldsymbol{\Sigma}_{X \Theta} & \boldsymbol{\Sigma}_{X X}
\end{array}\right]^{-1}\left[\begin{array}{l}
\boldsymbol{\theta} \\
\mathbf{x}
\end{array}\right]+\frac{1}{2} \mathbf{x}^{T} \boldsymbol{\Sigma}_{X X}^{-1} \mathbf{x}\right\}}_{H(\boldsymbol{\theta}, \mathbf{x})} .
\end{aligned}
$$

The tedious task here is to simplify $H(\boldsymbol{\theta}, \mathbf{x})$.

Regardless of what the 2-by-2 matrix inverse is, the matrix will take the form

$$
\left[\begin{array}{ll}
\boldsymbol{\Sigma}_{\Theta \Theta} & \boldsymbol{\Sigma}_{\Theta X} \\
\boldsymbol{\Sigma}_{X \Theta} & \boldsymbol{\Sigma}_{X X}
\end{array}\right]^{-1}=\left[\begin{array}{ll}
\boldsymbol{A} & \boldsymbol{B} \\
\boldsymbol{C} & \boldsymbol{D}
\end{array}\right]
$$

for some choices of matrices $\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}$ and $\boldsymbol{D}$. Therefore, the function $H(\boldsymbol{\theta}, \mathbf{x})$ can be written as

$$
H(\boldsymbol{\theta}, \mathbf{x})=-\frac{1}{2}\left\{\boldsymbol{\theta}^{T} \boldsymbol{A} \boldsymbol{\theta}+\boldsymbol{\theta}^{T} \boldsymbol{B} \mathbf{x}+\mathbf{x}^{T} \boldsymbol{C} \boldsymbol{\theta}+\mathbf{x}^{T} \boldsymbol{D} \mathbf{x}-\mathbf{x}^{T} \boldsymbol{\Sigma}_{X X}^{-1} \mathbf{x}\right\} .
$$

Our goal is to complete the square for $H(\boldsymbol{\theta}, \mathbf{x})$. To this end, we propose to write

$$
H(\boldsymbol{\theta}, \mathbf{x})=-\frac{1}{2}\left\{(\boldsymbol{\theta}-\boldsymbol{G} \mathbf{x})^{T} \boldsymbol{A}(\boldsymbol{\theta}-\boldsymbol{G} \mathbf{x})+Q(\mathbf{x})\right\}
$$

for some matrix $\boldsymbol{G}$ and function $Q(\cdot)$ of $\mathbf{x}$ only. If we compare Equation $(8.72)$ and Equation 8.73 , we observe that $\boldsymbol{G}$ must satisfy

$$
\boldsymbol{G}=-\boldsymbol{A}^{-1} \boldsymbol{B}
$$

Therefore, if we can determine $\boldsymbol{A}$ and $\boldsymbol{B}$, we will know $\boldsymbol{G}$. If we know $\boldsymbol{G}$, we have completed the square for $H(\boldsymbol{\theta}, \mathbf{x})$. If we can complete the square for $H(\boldsymbol{\theta}, \mathbf{x})$, we can write

$$
f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})=\underbrace{\frac{\exp \{-Q(\mathbf{x}) / 2\}}{\sqrt{(2 \pi)^{d}|\boldsymbol{\Sigma}| /\left|\boldsymbol{\Sigma}_{X X}\right|}}}_{\text {constant in } \boldsymbol{\theta}} \times \underbrace{\exp \left\{-\frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{G} \mathbf{x})^{T} \boldsymbol{A}(\boldsymbol{\theta}-\boldsymbol{G} \mathbf{x})\right\}}_{\text {a Gaussian }} .
$$

Hence, the MMSE estimate, which is the posterior mean $\mathbb{E}[\boldsymbol{\Theta} \mid \mathbf{X}=\mathbf{x}]$, is simply $\boldsymbol{G} \mathbf{x}$ :

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\operatorname{MMSE}}(\mathbf{x}) & =\mathbb{E}[\boldsymbol{\Theta} \mid \mathbf{X}=\mathbf{x}] \\
& =\boldsymbol{G} \mathbf{x} \\
& =-\boldsymbol{A}^{-1} \boldsymbol{B} \mathbf{x} .
\end{aligned}
$$



\subsection{MINIMUM MEAN-SQUARE ESTIMATION}

So it remains to determine $\boldsymbol{A}$ and $\boldsymbol{B}$ by solving the tedious matrix inversion problem. The result is 6

$$
\begin{aligned}
& \boldsymbol{A}=\left(\boldsymbol{\Sigma}_{\Theta \Theta}-\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1} \boldsymbol{\Sigma}_{X \Theta}\right)^{-1}, \\
& \boldsymbol{B}=-\left(\boldsymbol{\Sigma}_{\Theta \Theta}-\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1} \boldsymbol{\Sigma}_{X \Theta}\right)^{-1} \boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1}, \\
& \boldsymbol{C}=\left(\boldsymbol{\Sigma}_{X X}-\boldsymbol{\Sigma}_{X \Theta} \boldsymbol{\Sigma}_{\Theta \Theta}^{-1} \boldsymbol{\Sigma}_{\Theta X}\right)^{-1} \boldsymbol{\Sigma}_{X \Theta} \boldsymbol{\Sigma}_{\Theta \Theta}^{-1}, \\
& \boldsymbol{D}=\left(\boldsymbol{\Sigma}_{X X}-\boldsymbol{\Sigma}_{X \Theta} \boldsymbol{\Sigma}_{\Theta \Theta}^{-1} \boldsymbol{\Sigma}_{\Theta X}\right)^{-1} .
\end{aligned}
$$

Therefore, plugging everything into the equation,

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\mathrm{MMSE}}(\mathbf{x}) & =-\boldsymbol{A}^{-1} \boldsymbol{B} \mathbf{x} \\
& =\boldsymbol{\Sigma}_{\Theta, X} \boldsymbol{\Sigma}_{X X}^{-1} \mathbf{x} .
\end{aligned}
$$

For non-zero means, we can repeat the same arguments above and show that

$$
\widehat{\boldsymbol{\theta}}_{\mathrm{MMSE}}(\mathbf{x})=\boldsymbol{\mu}_{\Theta}+\boldsymbol{\Sigma}_{\Theta, X} \boldsymbol{\Sigma}_{X X}^{-1}\left(\mathbf{x}-\boldsymbol{\mu}_{X}\right)
$$

End of the proof. Please join us again.

Practice Exercise 8.9. Suppose $\boldsymbol{\Theta} \in \mathbb{R}^{d}$ and $\mathbf{X} \in \mathbb{R}^{N}$ are jointly Gaussian with a joint PDF

$$
\left[\begin{array}{l}
\boldsymbol{\Theta} \\
\mathbf{X}
\end{array}\right] \sim \operatorname{Gaussian}\left(\left[\begin{array}{c}
\boldsymbol{\mu}_{\Theta} \\
\boldsymbol{\mu}_{X}
\end{array}\right],\left[\begin{array}{cc}
\boldsymbol{\Sigma}_{\Theta \Theta} & \boldsymbol{\Sigma}_{\Theta X} \\
\boldsymbol{\Sigma}_{X \Theta} & \boldsymbol{\Sigma}_{X X}
\end{array}\right]\right) .
$$

We know that the MMSE estimator is

$$
\widehat{\boldsymbol{\Theta}}_{\operatorname{MMSE}}(\mathbf{X})=\boldsymbol{\mu}_{\Theta}+\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1}\left(\mathbf{X}-\boldsymbol{\mu}_{X}\right)
$$

Find the mean squared error when using the MMSE estimator.

Solution. Conditioned on $\mathbf{X}=\mathbf{x}$, according to Equation (8.70), the MMSE is

$$
\operatorname{MSE}(\boldsymbol{\Theta}, \widehat{\boldsymbol{\Theta}}(\mathbf{X}))=\operatorname{Tr}\{\operatorname{Cov}[\boldsymbol{\Theta} \mid \mathbf{X}]\}
$$

The conditional covariance $\operatorname{Cov}[\boldsymbol{\Theta} \mid \mathbf{X}]$ is the covariance of the posterior distribution $f_{\Theta \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$, which is

$$
\begin{aligned}
\operatorname{Tr}\{\operatorname{Cov}[\boldsymbol{\Theta} \mid \mathbf{X}]\} & =\operatorname{Tr}\{\boldsymbol{A}\} \\
& =\operatorname{Tr}\left\{\left(\boldsymbol{\Sigma}_{\Theta \Theta}-\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1} \boldsymbol{\Sigma}_{X \Theta}\right)^{-1}\right\}
\end{aligned}
$$

${ }^{6}$ See Matrix Cookbook https://www.math.uwaterloo.ca/ hwolkowi/matrixcookbook.pdf Section 9.1.5 on the Schur complement.

\section{CHAPTER 8. ESTIMATION}

The overall mean squared error is

$$
\begin{aligned}
\operatorname{MSE}(\boldsymbol{\Theta}, \widehat{\boldsymbol{\Theta}}(\cdot)) & =\mathbb{E}_{\mathbf{X}}[\operatorname{MSE}(\boldsymbol{\Theta}, \widehat{\boldsymbol{\Theta}}(\mathbf{X}))] \\
& =\int \operatorname{MSE}(\boldsymbol{\Theta}, \widehat{\boldsymbol{\Theta}}(\mathbf{x})) f_{\mathbf{X}}(\mathbf{x}) d \mathbf{x} \\
& =\int \operatorname{Tr}\{\operatorname{Cov}[\boldsymbol{\Theta} \mid \mathbf{X}]\} f_{\mathbf{X}}(\mathbf{x}) d \mathbf{x} \\
& =\int \operatorname{Tr}\left\{\left(\boldsymbol{\Sigma}_{\Theta \Theta}-\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1} \boldsymbol{\Sigma}_{X \Theta}\right)^{-1}\right\} f_{\mathbf{X}}(\mathbf{x}) d \mathbf{x} \\
& =\operatorname{Tr}\left\{\left(\boldsymbol{\Sigma}_{\Theta \Theta}-\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1} \boldsymbol{\Sigma}_{X \Theta}\right)^{-1}\right\} \int f_{\mathbf{X}}(\mathbf{x}) d \mathbf{x} \\
& =\operatorname{Tr}\left\{\left(\boldsymbol{\Sigma}_{\Theta \Theta}-\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1} \boldsymbol{\Sigma}_{X \Theta}\right)^{-1}\right\}
\end{aligned}
$$

For multidimensional Gaussian, does MMSE = MAP?

The answer is $Y E S$.

Theorem 8.4. Suppose $\boldsymbol{\Theta} \in \mathbb{R}^{d}$ and $\mathbf{X} \in \mathbb{R}^{N}$ are jointly Gaussian with a joint PDF

$$
\left[\begin{array}{l}
\boldsymbol{\Theta} \\
\mathbf{X}
\end{array}\right] \sim \operatorname{Gaussian}\left(\left[\begin{array}{l}
\boldsymbol{\mu}_{\Theta} \\
\boldsymbol{\mu}_{X}
\end{array}\right],\left[\begin{array}{ll}
\boldsymbol{\Sigma}_{\Theta \Theta} & \boldsymbol{\Sigma}_{\Theta X} \\
\boldsymbol{\Sigma}_{X \Theta} & \boldsymbol{\Sigma}_{X X}
\end{array}\right]\right)
$$

The MAP estimate is

$$
\widehat{\boldsymbol{\Theta}}_{M A P}(\mathbf{X})=\boldsymbol{\mu}_{\Theta}+\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1}\left(\mathbf{X}-\boldsymbol{\mu}_{X}\right)
$$

Proof. The proof of this result is straightforward. If we return to the proof of the MMSE result, we note that

$$
f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})=\underbrace{\frac{\exp \{-Q(\mathbf{x}) / 2\}}{\sqrt{(2 \pi)^{d}|\boldsymbol{\Sigma}| /\left|\boldsymbol{\Sigma}_{X X}\right|}}}_{\text {constant in } \boldsymbol{\theta}} \times \underbrace{\exp \left\{-\frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{G} \mathbf{x})^{T} \boldsymbol{A}(\boldsymbol{\theta}-\boldsymbol{G} \mathbf{x})\right\}}_{\text {a Gaussian }} .
$$

Therefore, the maximizer of this posterior distribution, which is the MAP estimate, is

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}(\mathbf{x}) & =\underset{\boldsymbol{\theta}}{\operatorname{argmax}} f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x}) \\
& =\underset{\boldsymbol{\theta}}{\operatorname{argmax}}-\frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{G} \mathbf{x})^{T} \boldsymbol{A}(\boldsymbol{\theta}-\boldsymbol{G} \mathbf{x}) .
\end{aligned}
$$

Taking the derivative w.r.t. $\boldsymbol{\theta}$ and setting it zero, we have

$$
\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}(\mathbf{x})=\boldsymbol{G} \mathbf{x}=\boldsymbol{\Sigma}_{\Theta, X} \boldsymbol{\Sigma}_{X X}^{-1} \mathbf{x} .
$$

If the mean vectors are non-zero, we have $\widehat{\boldsymbol{\theta}}_{\mathrm{MAP}}(\mathbf{x})=\boldsymbol{\mu}_{\Theta}+\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1}\left(\mathbf{x}-\boldsymbol{\mu}_{X}\right)$.

\subsection{MINIMUM MEAN-SQUARE ESTIMATION}

\subsubsection{Linking MMSE and neural networks}

The blossoming of deep neural networks since 2010 has created a substantial impact on modern data science. The basic idea of a neural network is to train a stack of matrices and nonlinear functions (known as the network weights and the neuron activation functions, respectively), among other innovative ideas, so that a certain training loss is minimized. Expressing this by equations, the goal of the learning is equivalent to solving the optimization problem

$$
\widehat{g}(\cdot)=\underset{g(\cdot)}{\operatorname{argmin}} \mathbb{E}_{\mathbf{X}, \boldsymbol{\Theta}}\left[\|\boldsymbol{\Theta}-g(\mathbf{X})\|^{2}\right],
$$

where $\mathbf{X} \in \mathbb{R}^{M}$ is the input data and $\boldsymbol{\Theta} \in \mathbb{R}^{d}$ is the ground truth prediction. We want to find $g(\cdot)$ such that the error is minimized.

The error we choose here is the $\ell_{2}$-norm error $\|\cdot\|^{2}$. It is only one of many possible choices. You may recognize that this is exactly the same as the MMSE optimization. Therefore, the neural network we are finding here is the MMSE estimator. Since the MMSE estimator is the conditional expectation of the posterior distribution, the neural network approximates the mean of the posterior distribution.

Often the struggle we have with deep neural networks is whether we can find the optimal network parameters via optimization algorithms such as the stochastic gradient descent algorithms. However, if we think about this problem more deeply, the equivalence between the MMSE estimator and the posterior mean tells us that the hard part is related to the posterior distribution. In the high-dimensional landscape, it is close to impossible to determine the posterior and its mean. If we add to these difficulties and the nonconvexity of the function $g$, training a network is very challenging.

One misconception about neural networks is that if we can achieve a low training error, and if the model can also achieve a low testing error, then the network is good. This is a false sense of satisfaction. If a model can achieve very good training and testing errors, then the model is only good with respect to the error you choose. For example, if we choose the $\ell_{2^{-}}$ norm error $\|\cdot\|^{2}$ and if our model achieves good training and testing errors (in terms of $\|\cdot\|^{2}$ ), we can conclude that the model does well with respect to $\|\cdot\|^{2}$. The more serious problem here, unfortunately, is that $\|\cdot\|^{2}$ is not necessarily a good metric of performance (for both training and testing) because training with $\|\cdot\|^{2}$ is equivalent to approximating the posterior mean. There is absolutely no reason to believe that in the high-dimensional landscape, the posterior mean is the optimal. If we choose the posterior mode or the posterior median, we will also obtain a result. Why are the modes and medians "worse" than the mean? In practice, it has been observed that training deep neural networks for image-processing tasks generally leads to over-smoothed images. This demonstrates how minimizing the mean squared error $\|\cdot\|^{2}$ can be a fundamental mismatch with the problem.

Is minimizing the MSE the best option?

- No. Minimizing the MSE is equivalent to finding the mean of the posterior. There is no reason why the mean is the "best".

- You can find the mode of the posterior, in which case you will get a MAP estimator.

- You can also find the median of the posterior, in which case you will get the minimum absolute error estimator.

\section{CHAPTER 8. ESTIMATION}

- Ultimately, you need to define what is "good" and what is "bad".

- The same principle applies to deep neural networks. Especially in the regression setting, why is $\|\cdot\|^{2}$ a good evaluation metric for testing (not just training)?

\subsection{Summary}

In this chapter, we have discussed the basic principles of parameter estimation. The three building blocks are:

- Likelihood $f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})$ : the PDF that we observe samples $\mathbf{X}$ conditioned on the unknown parameter $\boldsymbol{\Theta}$. In the frequentist world, $\boldsymbol{\Theta}$ is a deterministic quantity. In the Bayesian world, $\Theta$ is random and so it has a PDF.

- Prior $f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})$ : the PDF of $\boldsymbol{\Theta}$. The prior $f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})$ is used by all Bayesian computation.

- Posterior $f_{\boldsymbol{\Theta} \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$ : the PDF that the underlying parameter is $\boldsymbol{\Theta}=\boldsymbol{\theta}$ given that we have observed $\mathbf{X}=\mathbf{x}$.

The three building blocks give us several strategies to estimate the parameters:

- Maximum likelihood $(\mathrm{ML})$ estimation: Maximize $f_{\mathbf{X} \mid \Theta}(\mathbf{x} \mid \boldsymbol{\theta})$.

- Maximum a posteriori (MAP) estimation: Maximize $f_{\Theta \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$.

- Minimum mean-square estimation (MMSE): Minimize the mean squared error, which is equivalent to finding the mean of $f_{\Theta \mid \mathbf{X}}(\boldsymbol{\theta} \mid \mathbf{x})$.

As discussed in this chapter, no single estimation strategy is universally "better" because one needs to specify the optimality criterion. If the goal is to minimize the mean squared error, then the MMSE estimator is the optimal strategy. If the goal is to maximize the likelihood without assuming any prior knowledge, the ML estimator would be the optimal strategy. It may appear that if we knew the ground truth parameter $\boldsymbol{\theta}^{*}$ we could minimize the distance between the estimated parameter $\boldsymbol{\theta}$ and the true value $\boldsymbol{\theta}^{*}$. If the parameter is a scalar, this will work. However, if the parameter is a vector, the noise of the distance becomes an issue. For example, if one cares about the mean absolute error (MAE), the optimal estimator would be the median of the posterior distribution instead of the mean of the posterior in the MMSE case. Therefore, it is the end user's responsibility to specify the optimality criterion.

Whenever we consider parameter estimation, we tend to think that it is about estimating the model parameters, such as the mean of a Gaussian PDF. While in many statistics problems this is indeed the case, parameter estimation can be much broader if we link it with regression. Specifically, a regularized linear regression problem can be formulated as a MAP estimation

$$
\boldsymbol{\theta}^{*}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \underbrace{\|\mathbf{X} \boldsymbol{\theta}-\boldsymbol{y}\|^{2}}_{-\log f_{\mathbf{X} \mid \boldsymbol{\Theta}}(\mathbf{x} \mid \boldsymbol{\theta})}+\underbrace{\lambda R(\boldsymbol{\theta})}_{-\log f_{\boldsymbol{\Theta}}(\boldsymbol{\theta})},
$$

for some regularization $R(\boldsymbol{\theta})$, which is also the negative log of the prior. Expressed in this way, we recognize that the MAP estimation can be used to recover signals. For example, we

\subsection{REFERENCES}

can model $\mathbf{X}$ as a linear degradation process of certain imaging systems. Then solving the MAP estimation is equivalent to finding the best signal explaining the degraded observation using the posterior as the criterion. There is rich literature dealing with solving MAP estimation problems similar to these in subjects such as computational imaging, communication systems, remote sensing, radar engineering, and recommendation systems, to name a few.

\subsection{References}

\section{Basic}

8-1 Dimitri P. Bertsekas and John N. Tsitsiklis, Introduction to Probability, Athena Scientific, 2nd Edition, 2008. Chapter 8 and Chapter 9.

8-2 Alberto Leon-Garcia, Probability, Statistics, and Random Processes for Electrical Engineering, Prentice Hall, 3rd Edition, 2008. Chapter 6 and Chapter 8.

8-3 Athanasios Papoulis and S. Unnikrishna Pillai, Probability, Random Variables and Stochastic Processes, McGraw-Hill, 4th Edition, 2001. Chapter 8.

8-4 Henry Stark and John W. Woods, Probability and Random Processes with Applications to Signal Processing, Prentice Hall, 3rd Edition, 2002. Chapter 5.

8-5 Todd K. Moon and Wynn C. Stirling, Mathematical Methods and Algorithms for Signal Processing, Prentice-Hall, 2000. Chapter 12.

\section{Theoretical analysis}

8-6 H. Vincent Poor, An Introduction Signal Detection and Estimation, Springer, 1998.

8-7 Steven M. Kay, Fundamentals of Statistical Signal Processing: Estimation Theory, Prentice-Hall, 1993.

8-8 Bernard C. Levy, Principles of Signal Detection and Parameter Estimation, Springer, 2008.

8-9 Athanasios Papoulis and S. Unnikrishna Pillai, Probability, Random Variables and Stochastic Processes, McGraw-Hill, 2001. Chapter 8.

8-10 Larry Wasserman, All of Statistics: A Concise Course in Statistical Inference, Springer, 2010.

8-11 Erich L. Lehmann, Elements of Large-Sample Theory, Springer, 1999. Chapter 7.

8-12 George Casella and Roger L. Berger Statistical Inference, Duxbury, 2002. Chapter 7.

\section{CHAPTER 8. ESTIMATION}

\section{Machine-learning}

8-13 Christopher Bishop, Pattern Recognition and Machine Learning, Springer, 2006. Chapter 2 and Chapter 3.

8-14 Richard O. Duda, Peter E. Hart and David G. Stork, Pattern Classification, Wiley 2001. Chapter 3.

\subsection{Problems}

\section{Exercise 1.}

Let $x^{(1)}, \ldots, x^{(n)}$ be a sequence of i.i.d. Bernoulli random variables with $\mathbb{P}\left[x^{(n)}=1\right]=\theta$. Suppose that we have observed $x^{(1)}, \ldots, x^{(n)}$.

(a) Show that the PMF of $x^{(n)}$ is $p_{x^{(n)}}\left(x^{(n)} \mid \theta\right)=\theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}$. Find the joint PMF

$$
p_{x^{(1)}, \ldots, x^{(n)}}\left(x^{(1)}, \ldots, x^{(n)}\right)
$$

(b) Find the maximum likelihood estimate $\widehat{\theta}$, i.e.,

$$
\widehat{\theta}=\underset{\theta}{\operatorname{argmax}} \log p_{x^{(1)}, \ldots, x^{(n)}}\left(x^{(1)}, \ldots, x^{(n)}\right)
$$

Express your answer in terms of $x^{(1)}, \ldots, x^{(n)}$.

(c) Let $\theta=1 / 2$. Use Chebyshev's inequality to find an upper bound for $\mathbb{P}\left[\left|\widehat{\Theta}_{\mathrm{ML}}-\theta\right|>0.1\right]$.

\section{Exercise 2.}

Let $Y_{n}=\theta+W_{n}$ be the output of a noisy channel where the input is a scalar $\theta$ and $W_{n} \sim \mathcal{N}(0,1)$ is an i.i.d. Gaussian noise. Suppose that we have observed $y_{1}, \ldots, y_{N}$.

(a) Express the PDF of $Y_{n}$ in terms of $\theta$ and $y_{n}$. Find the joint PDF of $Y_{1}, \ldots, Y_{N}$.

(b) Find the maximum likelihood estimate $\widehat{\theta}$. Express your answer in terms of $y_{1}, \ldots, y_{N}$.

(c) Find $\mathbb{E}\left[\widehat{\Theta}_{\mathrm{ML}}\right]$

\section{Exercise 3.}

Let $x^{(1)}, \ldots, x^{(n)}$ be a sequence of i.i.d. Gaussian random variables with unknown mean $\theta_{1}$ and variance $\theta_{2}$. Suppose that we have observations $x^{(1)}, \ldots, x^{(n)}$.

(a) Express the PDF of $x^{(n)}$ in terms of $x^{(n)}, \theta_{1}$ and $\theta_{2}$. Find the joint $\mathrm{PDF}$ of $x^{(1)}, \ldots, x^{(n)}$.

(b) Find the maximum likelihood estimates of $\theta_{1}$ and $\theta_{2}$.

\subsection{PROBLEMS}

\section{Exercise 4.}

In this problem we study a single-photon image sensor. First, recall that photons arrive according to a Poisson distribution, i.e., the probability of observing $k$ photons is

$$
\mathbb{P}[Y=k]=\frac{\lambda^{k} e^{-\lambda}}{k !}
$$

where $\lambda$ is the (unknown) underlying photon arrival rate. When photons arrive at the singlephoton detector, the detector generates a binary response "1" when one or more photons are detected, and "0" when no photon is detected.

(a) Let $B$ be the random variable denoting the response of the single-photon detector. That is,

$$
B= \begin{cases}1, & Y \geq 1 \\ 0, & Y=0\end{cases}
$$

Find the PMF of $B$

(b) Suppose we have obtained $T$ independent measurements with realizations $B_{1}=b_{1}$, $B_{2}=b_{2}, \ldots, B_{T}=b_{T}$. Show that the underlying photon arrival rate $\lambda$ can be estimated by

$$
\lambda=-\log \left(1-\frac{\sum_{t=1}^{T} b_{t}}{T}\right) .
$$

(c) Get a random image from the internet and turn it into a grayscale array with values between 0 and 1. Write a MATLAB or Python program to synthetically generate a sequence of $T=1000$ binary images. Then use the previous result to reconstruct the grayscale image.

\section{Exercise 5.}

Consider a deterministic vector $s \in \mathbb{R}^{d}$ and random vectors

$$
\begin{aligned}
f_{\boldsymbol{Y} \mid \Theta}(\boldsymbol{y} \mid \theta) & =\operatorname{Gaussian}(\boldsymbol{s} \theta, \boldsymbol{\Sigma}), \\
f_{\Theta}(\theta) & =\operatorname{Gaussian}\left(\mu, \sigma^{2}\right) .
\end{aligned}
$$

(a) Show that the posterior distribution is given by

$$
f_{\boldsymbol{\Theta} \mid \boldsymbol{Y}}(\theta \mid \boldsymbol{y})=\operatorname{Gaussian}\left(m, q^{2}\right)
$$

where

$$
\begin{aligned}
d^{2} & =\boldsymbol{s}^{T} \boldsymbol{\Sigma}^{-1} \boldsymbol{s}, \\
m & =\left(d^{2}+\frac{1}{\sigma^{2}}\right)^{-1}\left(\boldsymbol{s}^{T} \boldsymbol{\Sigma}^{-1} \boldsymbol{y}+\frac{\mu}{\sigma^{2}}\right), \\
q^{2} & =\frac{1}{d^{2}+\frac{1}{\sigma^{2}}} .
\end{aligned}
$$

(b) Show that the MMSE estimate $\widehat{\theta}_{\operatorname{MMSE}}(\boldsymbol{y})$ is given by

$$
\widehat{\theta}_{\operatorname{MMSE}}(\boldsymbol{y})=\frac{\sigma^{2} \boldsymbol{s}^{T} \boldsymbol{\Sigma}^{-1} \boldsymbol{y}+\mu}{\sigma^{2} d^{2}+1}
$$



\section{CHAPTER 8. ESTIMATION}

(c) Show that the MSE is given by

$$
\operatorname{MSE}\left(\Theta, \widehat{\Theta}_{\operatorname{MMSE}}(\boldsymbol{Y})\right)=\frac{1}{d^{2}+\frac{1}{\sigma^{2}}}
$$

What happens when $\sigma \rightarrow 0$ ?

(d) Give an interpretation of $d^{2}$. What happens when $d^{2} \rightarrow 0$ and when $d^{2} \rightarrow \infty$ ?

\section{Exercise 6.}

Prove the following identity:

$$
\begin{aligned}
& {\left[\begin{array}{ll}
\boldsymbol{\Sigma}_{\Theta \Theta} & \boldsymbol{\Sigma}_{\Theta X} \\
\boldsymbol{\Sigma}_{X \Theta} & \boldsymbol{\Sigma}_{X X}
\end{array}\right]^{-1}} \\
& =\left[\begin{array}{cr}
\left(\boldsymbol{\Sigma}_{\Theta \Theta}-\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1} \boldsymbol{\Sigma}_{X \Theta}\right)^{-1} & -\left(\boldsymbol{\Sigma}_{\Theta \Theta}-\boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1} \boldsymbol{\Sigma}_{X \Theta}\right)^{-1} \boldsymbol{\Sigma}_{\Theta X} \boldsymbol{\Sigma}_{X X}^{-1} \\
\left(\boldsymbol{\Sigma}_{X X}-\boldsymbol{\Sigma}_{X \Theta} \boldsymbol{\Sigma}_{\Theta \Theta}^{-1} \boldsymbol{\Sigma}_{\Theta X}\right)^{-1} \boldsymbol{\Sigma}_{X \Theta} \boldsymbol{\Sigma}_{\Theta \Theta}^{-1} & \left(\boldsymbol{\Sigma}_{X X}-\boldsymbol{\Sigma}_{X \Theta} \boldsymbol{\Sigma}_{\Theta \Theta}^{-1} \boldsymbol{\Sigma}_{\Theta X}\right)^{-1}
\end{array}\right] .
\end{aligned}
$$

Hint: You can perform reverse engineering by checking whether the product of the left-hand side and the right-hand side would give you the identity matrix.

\section{Exercise 7.}

Let $x^{(1)}, x^{(2)}, X_{3}$ and $X_{4}$ be four i.i.d. Poisson random variables with mean $\theta=4$. Find the mean and variance of the following estimators $\widehat{\Theta}(\mathbf{X})$ for $\theta$ and determine whether they are biased or unbiased.

- $\widehat{\Theta}(\mathbf{X})=\left(x^{(1)}+x^{(2)}\right) / 2$

- $\widehat{\Theta}(\mathbf{X})=\left(X_{3}+X_{4}\right) / 2$

- $\widehat{\Theta}(\mathbf{X})=\left(x^{(1)}+2 x^{(2)}\right) / 3$

- $\widehat{\Theta}(\mathbf{X})=\left(x^{(1)}+x^{(2)}+X_{3}+X_{4}\right) / 4$

\section{Exercise 8.}

Let $x^{(1)}, \ldots, x^{(n)}$ be i.i.d. random variables with a uniform distribution of $[0, \theta]$. Consider the following estimator:

$$
\widehat{\Theta}(\mathbf{X})=\max \left(x^{(1)}, \ldots, x^{(n)}\right)
$$

(a) Show that the PDF of $\widehat{\Theta}$ is $f_{\widehat{\Theta}}(\theta)=N\left[F_{X}(x)\right]^{N-1} f_{X}(x)$, where $f_{X}$ and $F_{X}$ are respectively the PDF and CDF of $x^{(n)}$.

(b) Show that $\widehat{\Theta}$ is a biased estimator.

(c) Find the variance of $\widehat{\Theta}$. Is it a consistent estimator?

(d) Find a constant $c$ so that $c \widehat{\Theta}$ is unbiased.

\section{Exercise 9.}

Let $x^{(1)}, \ldots, x^{(n)}$ be i.i.d. Gaussian random variables with unknown mean $\theta$ and known variance $\sigma=1$.

\subsection{PROBLEMS}

(a) Show that the log-likelihood function is

$$
\log \mathcal{L}(\theta \mid \mathbf{x})=-\frac{N}{2} \log (2 \pi)-\frac{1}{2} \sum_{n=1}^{N}\left(x^{(n)}-\theta\right)^{2} .
$$

(b) Let $\overline{X^{2}}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}^{2}$ and $\bar{X}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}$. Show that $\overline{X^{2}}>(\bar{X})^{2}$ if and only if $\sum_{n=1}^{N}\left(x^{(n)}-\theta\right)^{2} \geq 0$ for all $\theta$.

(c) Use Python to plot the function $\log \mathcal{L}(\theta \mid \mathbf{x})$, when $\bar{X}=2$ and $\overline{X^{2}}=1$.

\section{Exercise 10.}

Let $x^{(1)}, \ldots, x^{(n)}$ be i.i.d. uniform random variables over the interval $[0, \theta]$.

Let $T=\max \left(x^{(1)}, \ldots, x^{(n)}\right)$.

(a) Consider the estimator $h(\mathbf{X})=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}$. Is $h(\cdot)$ an unbiased estimator?

(b) Consider the estimator $g(\mathbf{X})=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}$. Is $g(\cdot)$ an unbiased estimator?

(c) Show that

$$
\mathbb{E}[g(\mathbf{X}) \mid T=t]=\left(\frac{N+1}{N}\right) t
$$

(d) Let $\widehat{g}(\mathbf{X})=\mathbb{E}[g(\mathbf{X}) \mid T]=\left(\frac{N+1}{N}\right) T$. Show that

$$
\mathbb{E}\left[\widehat{g}(\mathbf{X})^{2}\right]=\left(\frac{(N+1)^{2}}{N(N+2)}\right) \theta^{2}
$$

(e) Show that

$$
\mathbb{E}\left[(\widehat{g}(\mathbf{X})-\theta)^{2}\right]=\left(\frac{1}{N(N+2)}\right) \theta^{2}
$$

\section{Exercise 11.}

The Kullback-Leibler divergence between two distributions $p_{1}(\mathbf{x})$ and $p_{2}(\mathbf{x})$ is defined as

$$
\mathrm{KL}\left(p_{1} \| p_{2}\right)=\int p_{1}(\mathbf{x}) \log \frac{p_{1}(\mathbf{x})}{p_{2}(\mathbf{x})} d \mathbf{x}
$$

Suppose we approximate $p_{1}$ using a distribution $p_{2}$. Let us choose $p_{2}=\operatorname{Gaussian}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. Show that $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$, which minimize the KL divergence, are such that

$$
\boldsymbol{\mu}=\mathbb{E}_{\mathbf{x} \sim p_{1}(\mathbf{x})}[\mathbf{x}] \quad \text { and } \quad \boldsymbol{\Sigma}=\mathbb{E}_{\mathbf{x} \sim p_{1}(\mathbf{x})}\left[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^{T}\right]
$$

\section{Exercise 12.}

(a) Recall that the trace operator is defined as $\operatorname{tr}[\boldsymbol{A}]=\sum_{i=1}^{d}[\boldsymbol{A}]_{i, i}$. Prove the matrix identity

$$
\mathbf{x}^{T} \boldsymbol{A} \mathbf{x}=\operatorname{tr}\left[\boldsymbol{A} \mathbf{x} \mathbf{x}^{T}\right]
$$

where $\boldsymbol{A} \in \mathbb{R}^{d \times d}$.

\section{CHAPTER 8. ESTIMATION}

(b) Show that the likelihood function

$$
p(\mathcal{D} \mid \boldsymbol{\Sigma})=\prod_{n=1}^{N}\left\{\frac{1}{(2 \pi)^{d / 2}|\boldsymbol{\Sigma}|^{1 / 2}} \exp \left\{-\frac{1}{2}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)\right\}\right\}
$$

can be written as

$$
p(\mathcal{D} \mid \boldsymbol{\Sigma})=\frac{1}{(2 \pi)^{N d / 2}}\left|\boldsymbol{\Sigma}^{-1}\right|^{N / 2} \exp \left\{-\frac{1}{2} \operatorname{tr}\left[\boldsymbol{\Sigma}^{-1} \sum_{n=1}^{N}\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)\left(\mathbf{x}_{n}-\boldsymbol{\mu}\right)^{T}\right]\right\} .
$$

(c) Let $\boldsymbol{A}=\boldsymbol{\Sigma}^{-1} \widehat{\boldsymbol{\Sigma}}_{\mathrm{ML}}$, and $\lambda_{1}, \ldots, \lambda_{d}$ be the eigenvalues of $\boldsymbol{A}$. Show that the result from part (b) leads to

$$
p(\mathcal{D} \mid \boldsymbol{\Sigma})=\frac{1}{(2 \pi)^{N d / 2}\left|\widehat{\boldsymbol{\Sigma}}_{\mathrm{ML}}\right|^{N / 2}}\left(\prod_{i=1}^{d} \lambda_{i}\right)^{N / 2} \exp \left\{-\frac{N}{2} \sum_{i=1}^{d} \lambda_{i}\right\}
$$

Hint: For matrix $\boldsymbol{A}$ with eigenvalues $\lambda_{1}, \ldots, \lambda_{d}, \operatorname{tr}[\boldsymbol{A}]=\sum_{i=1}^{d} \lambda_{i}$.

(d) Find $\lambda_{1}, \ldots, \lambda_{d}$ such that Equation 8.87) is maximized.

(e) With the choice of $\lambda_{i}$ given in (d), derive the ML estimate $\widehat{\boldsymbol{\Sigma}}_{\mathrm{ML}}$.

(f) What would be the alternative way of finding $\widehat{\boldsymbol{\Sigma}}_{\mathrm{ML}}$ ? You do not need to prove it. Just briefly describe the idea.

(g) $\widehat{\boldsymbol{\Sigma}}_{\mathrm{ML}}$ is a biased estimate of the covariance matrix because $\mathbb{E}\left[\widehat{\boldsymbol{\Sigma}}_{\mathrm{ML}}\right] \neq \boldsymbol{\Sigma}$. Can you suggest an unbiased estimate $\widehat{\boldsymbol{\Sigma}}_{\text {unbias }}$ such that $\mathbb{E}\left[\widehat{\boldsymbol{\Sigma}}_{\text {unbias }}\right]=\boldsymbol{\Sigma}$ ? You don't need to prove it. Just state the result.

\section{Chapter 9}

\section{Confidence and Hypothesis}

In Chapters 7 and 8 we learned about regression and estimation, which allow us to determine the underlying parameters of our statistical models. After obtaining the estimates, we would like to quantify the accuracy of the estimates and draw statistical conclusions. Additionally, we would like to understand the confidence of these estimates along with their statistical significance. This chapter presents a few principles that involve analyzing the confidence of the estimates and conducting hypothesis testing. There are two main questions that we will address:

- How good is our estimate? This is a fundamental question about the estimator $\widehat{\Theta}$, a random variable with a $\mathrm{PDF}$, a mean, and a variance ${ }^{1}$ The estimator we construct today may be different from the estimator we construct tomorrow due to variations in the observed data. Therefore, the quality of the estimator depends on the randomness and the number of samples used to construct it. To measure the quality of the estimator we need to introduce an important concept known as the confidence.

- Is there statistical significance? Suppose that we ran a campaign and observed that there is a change in the statistics. On what basis do we claim that the change is statistically significant? How should the cutoff be determined? If we claim that a result is statistically significant but there is no significance in reality, how much error will we suffer? These questions are the subjects of hypothesis testing.

These two principal questions are critical for modern data science. If they are not properly answered, our statistical conclusions could potentially be flawed. A toy example:

Imagine that you are developing a COVID-19 vaccine. You tested the vaccine on three patients, and all of them show positive responses to the vaccine. You felt excited because your vaccine has a $100 \%$ success rate. You submit your vaccine application to FDA. Within 1 second your application is rejected. Why? The answer is obvious. You only have three testing samples. How reliable can these three samples be?

While you are laughing at this toy example, it raises deep statistical questions. First, why are three samples not enough? Well, it is because the variance of the estimator can potentially be huge. More samples are better because if the estimator is the sample average of the individual responses, the estimator will behave like a Gaussian according to the Central

${ }^{1}$ Not all random variables have a well-defined PDF, mean, and variance. E.g., a Cauchy variable does not have a mean.