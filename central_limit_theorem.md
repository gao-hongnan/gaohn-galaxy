

\subsection{Central Limit Theorem}

The law of large numbers tells us the mean of the sample average $\bar{X}=(1 / N) \sum_{n=1}^{N} X_{n}$. However, if you recall our experiment of throwing $N$ dice and inspecting the PDF of the sum of the numbers, you may remember that the convolution of an infinite number of uniform distributions gives us a Gaussian distribution. For example, we show a sequence of experiments in Figure 6.15. In each experiment, we throw $N$ dice and count the sum. Therefore, if each face of the die is denoted as $X_{n}$, then the sum is $X_{1}+\cdots+X_{N}$. We plot the PDF of the sum. As you can see in the figure, $X_{1}+\cdots+X_{N}$ converges to a Gaussian. This phenomenon is explained by the Central Limit Theorem (CLT).

What does the Central Limit Theorem say? Let $\bar{X}$ be the sample average, and let $Z_{N}=\sqrt{N}\left(\frac{\bar{X}-\mu}{\sigma}\right)$ be the normalized variable. The Central Limit Theorem is as follows:

\subsection{CENTRAL LIMIT THEOREM}
![](https://cdn.mathpix.com/cropped/2023_02_15_18ba4dd3df9dc0499a5bg-49.jpg?height=414&width=1392&top_left_y=234&top_left_x=208)

Figure 6.15: Pictorial illustration of the Central Limit Theorem. Suppose we throw a die and record the face. [Left] If we only have one die, then the distribution of the face is uniform. [Middle] If we throw two dice, the distribution is the convolution of two uniform distributions. This will give us a triangle distribution. [Right] If we throw five dice, the distribution is becoming similar to a Gaussian. The Central Limit Theorem says that as $N$ goes to infinity, the distribution of the sum will converge to a Gaussian.

\section{Central Limit Theorem:}

The $\operatorname{CDF}$ of $Z_{N}$ is converging pointwise to the $\operatorname{CDF}$ of $\operatorname{Gaussian}(0,1)$.

Note that we are very careful here. We are not saying that the PDF of $Z_{N}$ is converging to the PDF of a Gaussian, nor are we saying that the random variable $Z_{N}$ is converging to a Gaussian random variable. We are only saying that the values of the CDF are converging pointwise. The difference is subtle but important.

To understand the difficulty and the core ideas, we first present the concept of convergence in distribution.

\subsubsection{Convergence in distribution}

Definition 6.8. Let $Z_{1}, \ldots, Z_{N}$ be random variables with $C D F s F_{Z_{1}}, \ldots, F_{Z_{N}}$ respectively. We say that a sequence of $Z_{1}, \ldots, Z_{N}$ converges in distribution to a random variable $Z$ with $C D F F_{Z}$ if

$$
\lim _{N \rightarrow \infty} F_{Z_{N}}(z)=F_{Z}(z),
$$

for every continuous point $z$ of $F_{Z}$. We write $Z_{N} \stackrel{d}{\rightarrow} Z$ to denote convergence in distribution.

This definition involves many concepts, which we will discuss one by one. However, the definition can be summarized in a nutshell as follows.

Example 1. (Bernoulli) Consider flipping a fair coin $N$ times. Denote each coin flip as a Bernoulli random variable $X_{n} \sim \operatorname{Bernoulli}(p)$, where $n=1,2, \ldots, N$. Define $Z_{N}$ as the sum

\section{CHAPTER 6. SAMPLE STATISTICS}

of $N$ Bernoulli random variables, so that

$$
Z_{N}=\sum_{n=1}^{N} X_{n} .
$$

We know that the resulting random variable $Z_{N}$ is a binomial random variable with mean $N p$ and variance $N p(1-p)$. Let us plot the $\operatorname{PDF} f_{Z_{N}}(z)$ as shown in Figure $6.16$
![](https://cdn.mathpix.com/cropped/2023_02_15_18ba4dd3df9dc0499a5bg-50.jpg?height=372&width=1132&top_left_y=568&top_left_x=273)

$$
\max _{z}\left|f_{Z_{N}}(z)-f_{Z}(z)\right| \quad \forall \rightarrow 0
$$

Figure 6.16: Convergence in distribution. The convergence in distribution concerns the convergence of the values of the CDF (not the PDF). In this figure, we let $Z_{N}=X_{1}+\cdots+X_{N}$, where $X_{N}$ is a Bernoulli random variable with parameter $p$. Since a sum of Bernoulli random variables is a binomial, $Z_{N}$ is a binomial random variable with parameters $(N, p)$. We plot the PDF of $Z_{N}$, which is a train of delta functions, and compare it with the Gaussian PDF. Observe that the error, $\max _{z}\left|f_{Z_{N}}(z)-f_{Z}(z)\right|$, does not converge to 0 . The PDF of $Z_{N}$ is a binomial. A binomial is always a binomial. It will not turn into a Gaussian.

The first thing we notice in the figure is that as $N$ increases, the PDF of the binomial has an envelope that is "very Gaussian". So one temptation is to say that the random variable $Z_{N}$ is converging to another random variable $Z$. In addition, we would think that the PDFs converge in the sense that for all $z$,

$$
f_{Z_{N}}(z)=\left(\begin{array}{c}
N \\
z
\end{array}\right) p^{z}(1-p)^{N-z} \longrightarrow f_{Z}(z)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{(z-\mu)^{2}}{2 \sigma^{2}}\right\},
$$

where $\mu=N p$ and $\sigma^{2}=N p(1-p)$.

Unfortunately this argument does not work, because $f_{Z}(z)$ is continuous but $f_{Z_{N}}(z)$ is discrete. The sample space of $Z_{N}$ and the sample space of $Z$ are completely different. In fact, if we write $f_{Z_{N}}$ as an impulse train, we observe that

$$
f_{Z_{N}}(z)=\sum_{i=0}^{N}\left(\begin{array}{c}
N \\
i
\end{array}\right) p^{i}(1-p)^{N-i} \delta(z-i)
$$

Clearly, no matter how big the $N$ is, the difference $\left|f_{Z_{N}}(z)-f_{Z}(z)\right|$ will never go to zero for non-integer values of $z$. Mathematically, we can show that

$$
\max _{z}\left|f_{Z_{N}}(z)-f_{Z}(z)\right| \nrightarrow 0,
$$

as $N \rightarrow \infty . Z_{N}$ is a binomial random variable regardless of $N$. It will not become a Gaussian.

\subsection{CENTRAL LIMIT THEOREM}

If $f_{Z_{N}}(z)$ is not converging to a Gaussian PDF, how do we explain the convergence? The answer is to look at the CDF. For discrete PDFs such as a binomial random variable, the CDF is a staircase function. What we can show is that

$$
F_{Z_{N}}(z)=\sum_{i=0}^{z}\left(\begin{array}{c}
N \\
i
\end{array}\right) p^{i}(1-p)^{N-i} \longrightarrow F_{Z}(z)=\int_{-\infty}^{z} \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{(t-\mu)^{2}}{2 \sigma^{2}}\right\} d t .
$$

The difference between the PDF convergence and the CDF convergence is that the PDF does not allow a meaningful "distance" between a discrete function and continuous function. For CDF, the distance is well defined by taking the difference between the staircase function and the continuous function. For example, we can compute

$$
\left|F_{Z_{N}}(z)-F_{Z}(z)\right|, \quad \text { for all continuous points } z \text { of } F_{Z},
$$

and show that

$$
\max _{z}\left|F_{Z_{N}}(z)-F_{Z}(z)\right| \longrightarrow 0
$$

![](https://cdn.mathpix.com/cropped/2023_02_15_18ba4dd3df9dc0499a5bg-51.jpg?height=510&width=1118&top_left_y=961&top_left_x=344)

Figure 6.17: Convergence in distribution. This is the same as Figure $6.16$ but this time we plot the CDF of $Z_{N}$. The CDF is a staircase function. We compare it with the Gaussian CDF. Observe that the error, $\max _{z}\left|F_{Z_{N}}(z)-F_{Z}(z)\right|$, converges to zero as $N$ grows. Convergence in distribution says that the sequence of CDFs $F_{Z_{N}}(z)$ will converge to the limiting $\operatorname{CDF} F_{Z}(z)$, at all continuous points of $F_{Z}(z)$.

We need to pay attention to the set of $z$ 's. We do not evaluate all $z$ 's but only the $z$ 's that are continuous points of $F_{Z}$. If $F_{Z}$ is Gaussian, this does not matter because all $z$ 's are continuous. However, for CDFs containing discontinuous points, our definition of convergence in distribution will ignore these discontinuous points because they have a measure zero.

Example 2. (Poisson) Consider $X_{n} \sim \operatorname{Poisson}(\lambda)$, and consider $X_{1}, \ldots, X_{N}$. Define $Z_{N}=$ $\sum_{n=1}^{N} X_{n}$. It follows that $\mathbb{E}\left[Z_{N}\right]=\sum_{n=1}^{N} \mathbb{E}\left[X_{n}\right]=N \lambda$ and $\operatorname{Var}\left[Z_{N}\right]=\sum_{n=1}^{N} \operatorname{Var}\left[X_{n}\right]=N \lambda$. Moreover, we know that the sum of Poissons remains a Poisson. Therefore, the PDF of $Z_{N}$ is

$$
f_{Z_{N}}(z)=\sum_{k=0}^{\infty} \frac{(N \lambda)^{k}}{k !} e^{-N \lambda} \delta(z-k) \quad \text { and } \quad f_{Z}(z)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{(z-\mu)^{2}}{2 \sigma^{2}}\right\}
$$



\section{CHAPTER 6. SAMPLE STATISTICS}
![](https://cdn.mathpix.com/cropped/2023_02_15_18ba4dd3df9dc0499a5bg-52.jpg?height=638&width=1326&top_left_y=238&top_left_x=176)

(a) $N=4$

(b) $N=10$

(c) $N=50$

Figure 6.18: Convergence in distribution for a sum of Poisson random variables. Here we assume that $X_{1}, \ldots, X_{N}$ are i.i.d. Poisson with a parameter $\lambda$. We let $Z_{N}=\sum_{n=1}^{N} X_{n}$ be the sum, and compute the corresponding PDF (top row) and CDFs (bottom row). Just as with the binomial example, the PDFs of the Poisson do not converge but the CDFs of the Poisson converge to the CDF of a Gaussian.

where $\mu=N \lambda$ and $\sigma^{2}=N \lambda$. Again, $f_{Z_{N}}$ does not converge to $f_{Z}$. However, if we compare the CDF, we can see from Figure $6.18$ that the CDF of the Poisson is becoming better approximated by the Gaussian.

Interpreting "convergence in distribution". After seeing two examples, you should now have some idea of what "convergence in distribution" means. This concept applies to the CDFs. When we write

$$
\lim _{N \rightarrow \infty} F_{Z_{N}}(z)=F_{Z}(z),
$$

we mean that $F_{Z_{N}}(z)$ is converging to the value $F_{Z}(z)$, and this relationship holds for all the continuous $z$ 's of $F_{Z}$. It does not say that the random variable $Z_{N}$ is becoming another random variable $Z$.

$$
Z_{N} \stackrel{d}{\longrightarrow} Z \text { is equivalent to } \lim _{N \rightarrow \infty} F_{Z_{N}}(z)=F_{Z}(z) .
$$

Example 3. (Exponential) So far, we have studied the sum of discrete random variables. Now, let's take a look at continuous random variables. Consider $X_{n} \sim \operatorname{Exponential}(\lambda)$, and let $X_{1}, \ldots, X_{N}$ be i.i.d. copies. Define $Z_{N}=\sum_{n=1}^{N} X_{n}$. Then $\mathbb{E}\left[Z_{N}\right]=\sum_{n=1}^{N} \mathbb{E}\left[X_{n}\right]=N / \lambda$ and $\operatorname{Var}\left[Z_{N}\right]=\frac{N}{\lambda^{2}}$. How about the PDF of $Z_{N}$ ? Using the characteristic functions, we know that

$$
f_{X_{n}}(x)=\lambda e^{-\lambda x} \stackrel{\mathcal{F}}{\longleftrightarrow} \Phi_{X_{n}}(j \omega)=\frac{\lambda}{\lambda+j \omega} .
$$



\subsection{CENTRAL LIMIT THEOREM}

Therefore, the product is

$$
\begin{aligned}
\Phi_{Z_{N}}(j \omega) & =\prod_{n=1}^{N} \Phi_{X_{n}}(j \omega)=\frac{\lambda^{N}}{(\lambda+j \omega)^{N}}=\frac{\lambda^{N}}{(\lambda+j \omega)^{N}} \times \frac{(N-1) !}{(N-1) !} \\
& =\frac{\lambda^{N}}{(N-1) !} \cdot \frac{(N-1) !}{(\lambda+j \omega)^{N}} \stackrel{\mathcal{F}}{\longleftrightarrow} \frac{\lambda^{N}}{(N-1) !} z^{N-1} e^{-\lambda z}=f_{Z_{N}}(z) .
\end{aligned}
$$

This resulting PDF $f_{Z_{N}}(z)=\frac{\lambda^{N}}{(N-1) !} z^{N-1} e^{-\lambda z}$ is known as the Erlang distribution. The CDF of the Erlang distribution is

$$
\begin{aligned}
& F_{Z_{N}}(z)=\int_{-\infty}^{z} f_{Z_{N}}(t) d t \\
& =\int_{0}^{z} \frac{\lambda^{N}}{(N-1) !} t^{N-1} e^{-\lambda t} d t \\
& =\operatorname{Gamma} \text { function }(z, N) \text {, }
\end{aligned}
$$

where the last integral is known as the incomplete gamma function, evaluated at $z$.

Given all these, we can now compare the PDF and the CDF of $Z_{N}$ versus $Z$. Figure $\mathbf{6 . 1 9}$ shows the PDFs and the CDFs of $Z_{N}$ for various $N$ values. In this experiment we set $\lambda=1$. As we can see from the experiment, the Erlang distribution's PDF and CDF converge to a Gaussian. In fact, for continuous random variables such as exponential random variables, we indeed have the random variable $Z_{N}$ converging to the random variable $Z$. This is quite different from discrete random variables, where $Z_{N}$ does not converge to $Z$ but only $F_{Z_{N}}$ converges to $F_{Z}$.
![](https://cdn.mathpix.com/cropped/2023_02_15_18ba4dd3df9dc0499a5bg-53.jpg?height=642&width=1332&top_left_y=1324&top_left_x=232)

(a) $N=4$

(b) $N=10$

(c) $N=50$

Figure 6.19: Convergence in distribution for a sum of exponential random variables. Here we assume that $X_{1}, \ldots, X_{N}$ are i.i.d. exponentials with a parameter $\lambda$. We define $Z_{N}=\sum_{n=1}^{N} X_{n}$ be the sum. It is known that the sum of exponentials is an Erlang. We compute the corresponding PDF (top row) and CDFs (bottom row). Unlike the previous two examples, in this example we see that both PDFs and CDFs of the Erlang distribution are converging to a Gaussian.

\section{CHAPTER 6. SAMPLE STATISTICS}

Is $\stackrel{d}{\longrightarrow}$ stronger than $\stackrel{p}{\longrightarrow}$ ? Convergence in distribution is actually weaker than convergence in probability. Consider a continuous random variable $X$ with a symmetric PDF $f_{X}(x)$ such that $f_{X}(x)=f_{X}(-x)$. It holds that the PDF of $-X$ has the same PDF. If we define the sequence $Z_{N}=X$ if $N$ is odd and $Z_{N}=-X$ if $N$ is even, and let $Z=X$, then $F_{Z_{N}}(z)=F_{Z}(z)$ for every $z$ because the PDF of $X$ and $-X$ are identical. Therefore, $Z_{N} \stackrel{d}{\rightarrow} Z$. However, $Z_{N} \stackrel{p}{\rightarrow} Z$ because $Z_{N}$ oscillates between the random variables $X$ and $-X$. These two random variables are different (although they have the same CDF) because $\mathbb{P}[X=-X]=\mathbb{P}[\{\omega: X(\omega)=-X(\omega)\}]=\mathbb{P}[\{\omega: X(\omega)=0\}]=0$.

\subsubsection{Central Limit Theorem}

Theorem $6.19$ (Central Limit Theorem). Let $X_{1}, \ldots, X_{N}$ be i.i.d. random variables of mean $\mathbb{E}\left[X_{n}\right]=\mu$ and variance $\operatorname{Var}\left[X_{n}\right]=\sigma^{2}$. Also, assume that $\mathbb{E}\left[\left|X_{n}^{3}\right|\right]<\infty$. Let $\bar{X}=(1 / N) \sum_{n=1}^{N} X_{n}$ be the sample average, and let $Z_{N}=\sqrt{N}\left(\frac{\bar{X}-\mu}{\sigma}\right)$. Then

$$
\lim _{N \rightarrow \infty} F_{\bar{Z}_{N}}(z)=F_{Z}(z),
$$

where $Z=\operatorname{Gaussian}(0,1)$.

In plain words, the Central Limit Theorem says that the sample average (which is a random variable) has a CDF converging to the CDF of a Gaussian. Therefore, if we want to evaluate probabilities associated with the sample average, we can approximate the probability by the probability of a Gaussian.

As we discussed above, the Central Limit Theorem does not mean that the random variable $Z_{N}$ is converging to a Gaussian random variable, nor does it mean that the PDF of $Z_{N}$ is converging to the PDF of a Gaussian. It only means that the CDF of $Z_{N}$ is converging to the CDF of a Gaussian. Many people think that the Central Limit Theorem means "sample average converges to Gaussian". This is incorrect for the above reasons. However, it is not completely wrong. For continuous random variables where both PDF and CDF are continuous, we will not run into situations where the PDF is a train of delta functions. In this case, convergence in CDF can be translated to convergence in PDF.

The power of the Central Limit Theorem is that the result holds for any distribution of $X_{1}, \ldots, X_{N}$. That is, regardless of the distribution of $X_{1}, \ldots, X_{N}$, the CDF of $\bar{X}$ is approaching a Gaussian.

\section{Summary of the Central Limit Theorem}

- $X_{1}, \ldots, X_{N}$ are i.i.d. random variables, with mean $\mu$ and variance $\sigma^{2}$. They are not necessarily Gaussians.

- Define the sample average as $\bar{X}=(1 / N) \sum_{n=1}^{N} X_{n}$, and let $Z_{N}=\sqrt{N}\left(\frac{\bar{X}-\mu}{\sigma}\right)$.

- The Central Limit Theorem says $Z_{N} \stackrel{d}{\longrightarrow}$ Gaussian $(0,1)$. Equivalently, the theorem says that $N \bar{X} \stackrel{d}{\longrightarrow} \operatorname{Gaussian}\left(\mu, \sigma^{2}\right)$.

- So if we want to evaluate the probability of $\bar{X} \in \mathcal{A}$ for some set $\mathcal{A}$, we can

\subsection{CENTRAL LIMIT THEOREM}

approximate the probability by evaluating the Gaussian:

$$
\mathbb{P}\left[\bar{X} \in \mathcal{A}\right] \approx \int_{\mathcal{A}} \frac{1}{\sqrt{2 \pi\left(\sigma^{2} / N\right)}} \exp \left\{-\frac{(y-\mu)^{2}}{2\left(\sigma^{2} / N\right)}\right\} d y
$$

- CLT does not say that the PDF of $\bar{X}$ is becoming a Gaussian PDF.

- CLT only says that the CDF of $\bar{X}$ is becoming a Gaussian CDF.

If the set $\mathcal{A}$ is an interval, we can use the standard Gaussian CDF to compute the probability.

Corollary 6.3. Let $X_{1}, \ldots, X_{N}$ be i.i.d. random variables with mean $\mu$ and variance $\sigma^{2}$. Define the sample average as $\bar{X}=(1 / N) \sum_{n=1}^{N} X_{n}$. Then

$$
\mathbb{P}\left[a \leq \bar{X} \leq b\right] \approx \Phi\left(\sqrt{N} \frac{b-\mu}{\sigma}\right)-\Phi\left(\sqrt{N} \frac{a-\mu}{\sigma}\right)
$$

where $\Phi(z)=\int_{-\infty}^{z} \frac{1}{\sqrt{2 \pi}} e^{-\frac{x^{2}}{2}} d x$ is the CDF of the standard Gaussian.

Proof. By the Central Limit Theorem, we know that $\bar{X} \stackrel{d}{\longrightarrow} \operatorname{Gaussian}\left(\mu, \frac{\sigma^{2}}{N}\right)$. Therefore,

$$
\begin{aligned}
\mathbb{P}\left[a \leq \bar{X} \leq b\right] & \approx \int_{a}^{b} \frac{1}{\sqrt{2 \pi\left(\sigma^{2} / N\right)}} \exp \left\{-\frac{(y-\mu)^{2}}{2\left(\sigma^{2} / N\right)}\right\} d y \\
& =\int_{\sqrt{N} \frac{a-\mu}{\sigma}}^{\sqrt{N} \frac{b-\mu}{\sigma}} \frac{1}{\sqrt{2 \pi}} e^{-\frac{y^{2}}{2}} d y=\Phi\left(\sqrt{N} \frac{b-\mu}{\sigma}\right)-\Phi\left(\sqrt{N} \frac{a-\mu}{\sigma}\right) .
\end{aligned}
$$
![](https://cdn.mathpix.com/cropped/2023_02_15_18ba4dd3df9dc0499a5bg-55.jpg?height=502&width=1392&top_left_y=1482&top_left_x=206)

Figure 6.20: The Central Limit Theorem says that if we want to evaluate the probability $\mathbb{P}\left[a \leq \bar{X} \leq b\right]$, where $\bar{X}=(1 / N) \sum_{n=1}^{N} X_{n}$ is the sample average of i.i.d. random variables $X_{1}, \ldots, X_{N}$, we can approximate the probability by integrating the Gaussian PDF.

A graphical illustration of the CLT is shown in Figure 6.20, where we use a binomial random variable (which is the sum of i.i.d. Bernoulli) as an example. The CLT does not say

\section{CHAPTER 6. SAMPLE STATISTICS}

that the binomial random variable is becoming a Gaussian. It only says that the probability covered by the binomial can be approximated by the Gaussian.

The following proof of the Central Limit Theorem can be skipped if this is your first time reading the book.

Proof of the Central Limit Theorem. We now give a "proof" of the Central Limit Theorem. Technically speaking, this proof does not prove the convergence of the CDF as the theorem claims; it only proves that the moment-generating function converges. The actual proof of the CDF convergence is based on the Berry-Esseen Theorem, which is beyond the scope of this book. However, what we prove below is still useful because it gives us some intuition about why Gaussian is the limiting random variable we should consider in the first place.

Let $Z_{N}=\sqrt{N}\left(\frac{\bar{X}-\mu}{\sigma}\right)$. It follows that $\mathbb{E}\left[Z_{N}\right]=0$ and $\operatorname{Var}\left[Z_{N}\right]=1$. Therefore, if we can show that $Z_{N}$ is converging to a standard Gaussian random variable $Z \sim \operatorname{Gaussian}(0,1)$, then by the linear transformation property of Gaussian, $Y=\frac{\sigma}{\sqrt{N}} Z+\mu$ will be Gaussian $\left(\mu, \sigma^{2} / N\right)$.

Our proof is based on analyzing the moment-generating function of $Z_{N}$. In particular,

$$
M_{Z_{N}}(s) \stackrel{\text { def }}{=} \mathbb{E}\left[e^{s Z_{N}}\right]=\mathbb{E}\left[e^{s \sqrt{N}\left(\frac{\bar{x}_{N}-\mu}{\sigma}\right)}\right]=\prod_{n=1}^{N} \mathbb{E}\left[e^{\frac{s}{\sigma \sqrt{N}}\left(X_{n}-\mu\right)}\right] .
$$

Expanding the exponential term using the Taylor expansion (Chapter 1.2),

$$
\begin{aligned}
& \prod_{n=1}^{N} \mathbb{E}\left[e^{\frac{s}{\sigma \sqrt{N}}\left(X_{n}-\mu\right)}\right] \\
& =\prod_{n=1}^{N} \mathbb{E}\left[1+\frac{s}{\sigma \sqrt{N}}\left(X_{n}-\mu\right)+\frac{s^{2}}{2 \sigma^{2} N}\left(X_{n}-\mu\right)^{2}+\mathcal{O}\left(\frac{\left(X_{n}-\mu\right)^{3}}{\sigma^{3} N \sqrt{N}}\right)\right] \\
& =\prod_{n=1}^{N}\left[1+\frac{s}{\sigma \sqrt{N}} \mathbb{E}\left[X_{n}-\mu\right]+\frac{s^{2}}{2 \sigma^{2} N} \mathbb{E}\left[\left(X_{n}-\mu\right)^{2}\right]\right]=\left(1+\frac{s^{2}}{2 N}\right)^{N} .
\end{aligned}
$$

It remains to show that $\left(1+\frac{s^{2}}{2 N}\right)^{N} \rightarrow e^{s^{2} / 2}$. If we can show that, we have shown that the MGF of $Z_{N}$ is also the MGF of Gaussian $(0,1)$. To this end, we consider $\log (1+x)$. By the Taylor approximation, we have that

$$
\log (1+x) \approx \log (1)+\left(\left.\frac{d}{d x} \log x\right|_{x=1}\right) x+\left(\left.\frac{d^{2}}{d x^{2}} \log x\right|_{x=1}\right) \frac{x^{2}}{2}+\mathcal{O}\left(x^{3}\right) .
$$

Therefore, we have $\log \left(1+\frac{s^{2}}{2 N}\right) \approx \frac{s^{2}}{2 N}-\frac{s^{4}}{4 N^{2}}$. As $N \rightarrow \infty$, the limit becomes

$$
\lim _{N \rightarrow \infty} N \log \left(1+\frac{s^{2}}{2 N}\right) \approx \frac{s^{2}}{2}-\lim _{N \rightarrow \infty} \frac{s^{4}}{4 N}=\frac{s^{2}}{2},
$$

and so taking the exponential on both sides yields $\lim _{N \rightarrow \infty}\left(1+\frac{s^{2}}{2 N}\right)^{N}=e^{\frac{s^{2}}{2}}$. Therefore, we conclude that $\lim _{N \rightarrow \infty} M_{Z_{N}}(s)=e^{\frac{s^{2}}{2}}$, and so $Z_{N}$ is converging to a Gaussian.

\subsection{CENTRAL LIMIT THEOREM}

Limitation of our proof. The limitation of our proof lies in the issue of whether the integration and the limit are interchangeable:

$$
\begin{aligned}
\lim _{N \rightarrow \infty} M_{Z_{N}}(s) & =\lim _{N \rightarrow \infty}\left\{\int f_{Z_{N}}(z) e^{s z} d z\right\} \\
& \stackrel{?}{=} \int\left(\lim _{N \rightarrow \infty} f_{Z_{N}}(z)\right) e^{s z} d z
\end{aligned}
$$

If they were, then proving $\lim _{N \rightarrow \infty} M_{Z_{N}}(s)=M_{Z}(s)$ is sufficient to claim $f_{Z_{N}}(z) \rightarrow f_{Z}(z)$. However, we know that the latter is not true in general. For example, if $f_{Z_{N}}(z)$ is a train of delta functions, then the limit and the integration are not interchangeable.

Berry-Esseen Theorem. The formal way of proving the Central Limit Theorem is to prove the Berry-Esseen Theorem. The theorem states that

$$
\sup _{z \in \mathbb{R}}\left|F_{Z_{N}}(z)-F_{Z}(z)\right| \leq C \frac{\beta}{\sigma^{3} \sqrt{N}},
$$

where $\beta$ and $C$ are universal constants. Here, you can more or less treat the supremum operator as the maximum. The left-hand side represents the worst-case error of the CDF $F_{Z_{N}}$ compared to the limiting $\operatorname{CDF} F_{Z}$. The right-hand side involves several constants $C$, $\beta$, and $\sigma$, but they are fixed.

As $N$ goes to infinity, the right-hand side will converge to zero. Therefore, if we can prove this result, then we have proved the actual Central Limit Theorem. In addition, we have found the rate of convergence since the right-hand side tells us that the error drops at the rate of $1 / \sqrt{N}$, which is not particularly fast but is sufficient for our purpose. Unfortunately, proving the Berry-Esseen theorem is not easy. One of the difficulties, for example, is that one needs to deal with the infinite convolutions in the time domain or the frequency domain.

Interpreting our proof. If our proof is not completely valid, why do we mention it? For one thing, it provides us with some useful intuition. For most of the (well-behaving) random variables whose moments are finite, the exponential term in the moment-generating function can be truncated to the second-order polynomial. Since a second-order polynomial is a Gaussian, it naturally concludes that as long as we can perform such truncation the truncated random variable will be Gaussian.

To convince you that the Gaussian MGF is the second-order approximation to other MGFs, we use Bernoulli as an example. Let $X_{1}, \ldots, X_{N}$ be i.i.d. Bernoulli with a parameter $p$. Then the moment-generating function of $\bar{X}=(1 / N) \sum_{n=1}^{N} X_{n}$ would be:

$$
\begin{aligned}
M_{\bar{X}}(s) & =\mathbb{E}\left[e^{s \bar{X}}\right]=\mathbb{E}\left[e^{s \frac{1}{N} \sum_{n=1}^{N} X_{n}}\right]=\prod_{n=1}^{N} \mathbb{E}\left[e^{\frac{s}{N} X_{n}}\right] \\
& =\left(1-p+p e^{\frac{s}{N}}\right)^{N} \approx\left(1-p+p\left(1+\frac{s}{N}+\frac{s^{2}}{2 N^{2}}\right)\right)^{N} \\
& =\left(1+\frac{s p}{N}+\frac{s^{2} p}{2 N^{2}}\right)^{N} .
\end{aligned}
$$



\section{CHAPTER 6. SAMPLE STATISTICS}

Using the logarithmic approximation, it follows that

$$
\begin{aligned}
\log M_{\bar{X}}(s) & =N \log \left(1+\frac{s p}{N}+\frac{s^{2} p}{2 N^{2}}\right) \\
& \approx N\left(\frac{s p}{N}+\frac{s^{2} p}{2 N^{2}}\right)-\frac{N}{2}\left(\frac{s p}{N}+\frac{s^{2} p}{2 N^{2}}\right)^{2} \\
& \approx s p+\frac{s^{2} p(1-p)}{2 N} \stackrel{\text { def }}{=} \log M_{Y}(s)
\end{aligned}
$$

Taking the exponential on both sides, we have that

$$
M_{Y}(s)=\exp \left\{s p+\frac{s^{2} p(1-p)}{2 N}\right\},
$$

which is the MGF of a Gaussian random variable $Y \sim \operatorname{Gaussian}\left(p, \frac{p(1-p)}{N}\right)$.

Figure $6.21$ shows several MGFs. In each of the subfigures we plot the exact MGF $M_{\bar{X}}(s)=\left(1-p+p e^{\frac{s}{N}}\right)^{N}$ as a function of $s$. (The parameter $p$ in this example is $p=0.5$.) We vary the number $N$, and we inspect how the shape of $M_{\bar{X}}(s)$ changes. On top of the exact MGFs, we plot the Gaussian approximations $M_{Y}(s)=\exp \left\{s p+\frac{s^{2} p(1-p)}{2 N}\right\}$. According to our calculation, this Gaussian approximation is the second-order approximation to the exact MGF. The figures show the effect of the second-order approximation. For example, in (a) when $N=2$ the Gaussian is a quadratic approximation of the exact MGF. For (b) and (c), as $N$ increases, the approximation improves.

![](https://cdn.mathpix.com/cropped/2023_02_15_18ba4dd3df9dc0499a5bg-58.jpg?height=302&width=432&top_left_y=1282&top_left_x=194)

(a) $N=2$

![](https://cdn.mathpix.com/cropped/2023_02_15_18ba4dd3df9dc0499a5bg-58.jpg?height=300&width=430&top_left_y=1283&top_left_x=625)

(b) $N=4$

![](https://cdn.mathpix.com/cropped/2023_02_15_18ba4dd3df9dc0499a5bg-58.jpg?height=295&width=434&top_left_y=1288&top_left_x=1050)

(c) $N=10$

Figure 6.21: Explanation of the Central Limit Theorem using the function. In this set of plots, we show the MGF of the random variable $\bar{X}=(1 / N) \sum_{n=1}^{N} X_{n}$, where $X_{1}, \ldots, X_{N}$ are i.i.d. Bernoulli random variables. The exact MGF of $\bar{X}$ is the binomial, whereas the approximated MGF is the Gaussian. We observe that as $N$ increases, the Gaussian approximation to the exact MGF improves.

The reason why the second-order approximation works for Gaussian is that when $N$ increases, the higher order moments of $\bar{X}$ vanish and only the leading first two moments survive. The MGFs are becoming flat because $M_{Y}(s)=\exp \left\{s p+\frac{s^{2} p(1-p)}{2 N}\right\}$ converges to $\exp \{s p\}$ when $N \rightarrow \infty$. Taking the inverse Laplace transform, $M_{Y}(s)=\exp \{s p\}$ corresponds to a delta function. This makes sense because as $N$ grows, the variance of the $\bar{X}$ shrinks.

\subsection{CENTRAL LIMIT THEOREM}

\subsubsection{Examples}

Example 6.15. Prove the equivalence of a few statements.

- $\sqrt{N}\left(\frac{\bar{X}-\mu}{\sigma}\right) \stackrel{d}{\rightarrow} \operatorname{Gaussian}(0,1)$

- $\sqrt{N}\left(\bar{X}-\mu\right) \stackrel{d}{\rightarrow} \operatorname{Gaussian}\left(0, \sigma^{2}\right)$

- $\sqrt{N} \bar{X} \stackrel{d}{\rightarrow} \operatorname{Gaussian}\left(\mu, \sigma^{2}\right)$

Solution. The proof is based on the linear transformation property of Gaussian random variables. For example, if the first statement is true, then the second statement is also true because

$$
\begin{aligned}
\lim _{N \rightarrow \infty} F_{\sqrt{N}\left(\bar{X}-\mu\right)}(z) & =\lim _{N \rightarrow \infty} \mathbb{P}\left[\sqrt{N}\left(\bar{X}-\mu\right) \leq z\right] \\
& =\lim _{N \rightarrow \infty} \mathbb{P}\left[\sqrt{N}\left(\frac{\bar{X}-\mu}{\sigma}\right) \leq \frac{z}{\sigma}\right] \\
& =\int_{-\infty}^{z / \sigma} \frac{1}{\sqrt{2 \pi}} e^{-\frac{t^{2}}{2}} d t \\
& =\int_{-\infty}^{z} \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{t^{2}}{2 \sigma^{2}}} d t
\end{aligned}
$$

The other results can be proved similarly.

Example 6.16. Suppose $X_{n} \sim \operatorname{Poisson}(10)$ for $n=1, \ldots, N$, and let $\bar{X}$ be the sample average. Use the Central Limit Theorem to approximate $\mathbb{P}\left[9 \leq \bar{X} \leq 11\right]$ for $N=20$

Solution. We first show that

$$
\begin{aligned}
\mathbb{E}\left[\bar{X}\right] & =\mathbb{E}\left[\frac{1}{N} \sum_{n=1}^{N} X_{n}\right]=\frac{1}{N} \sum_{n=1}^{N} \mathbb{E}\left[X_{n}\right]=10, \\
\operatorname{Var}\left[\bar{X}\right] & =\frac{1}{N^{2}} \sum_{n=1}^{N} \operatorname{Var}\left[X_{n}\right]=\frac{1}{N} \operatorname{Var}\left[X_{n}\right]=\frac{10}{20}=\frac{1}{2} .
\end{aligned}
$$

Therefore, the Central Limit Theorem implies that $\bar{X} \stackrel{d}{\longrightarrow} \operatorname{Gaussian}\left(10, \frac{1}{2}\right)$. The probability is

$$
\begin{aligned}
\mathbb{P}\left[9 \leq \bar{X} \leq 11\right] & \approx \Phi\left(\frac{11-10}{\sqrt{1 / 2}}\right)-\Phi\left(\frac{9-10}{\sqrt{1 / 2}}\right) \\
& =\Phi\left(\frac{1}{\sqrt{0.5}}\right)-\Phi\left(-\frac{1}{\sqrt{0.5}}\right)=0.9214-0.0786=0.8427
\end{aligned}
$$



\section{CHAPTER 6. SAMPLE STATISTICS}

We can also do an exact calculation to verify our approximation. Let $S_{N}=$ $\sum_{n=1}^{N} X_{n}$ so that $\bar{X}=\frac{S_{N}}{N}$. Since a sum of Poisson remains a Poisson, it follows that

$$
S_{N} \sim \operatorname{Poisson}(10 N)=\operatorname{Poisson}(200) .
$$

Consequently,

$$
\begin{aligned}
\mathbb{P}\left[9 \leq \bar{X} \leq 11\right] & =\mathbb{P}\left[180 \leq S_{N} \leq 220\right] \\
& =\sum_{\ell=0}^{220} \frac{200^{\ell} e^{-200}}{\ell !}-\sum_{\ell=0}^{180} \frac{200^{\ell} e^{-200}}{\ell !}=0.9247-0.0822=0.8425
\end{aligned}
$$

Note that this is an exact calculation subject to numerical errors when evaluating the finite sums. The proximity to the Gaussian approximation shows the convenience of the Central Limit Theorem.

Example 6.17. Suppose you have collected $N=100$ data points from an unknown distribution. The only thing you know is that the true population mean is $\mu=500$ and the standard deviation is $\sigma=80$. (Note that this distribution is not necessarily a Gaussian.)

(a) Find the probability that the sample mean will be inside the interval $(490,510)$.

(b) Find an interval such that $95 \%$ of the sample average is covered.

Solution. To solve $($ a $)$, we note that $\bar{X} \stackrel{d}{\rightarrow} \operatorname{Gaussian}\left(500,\left(\frac{80}{\sqrt{100}}\right)^{2}\right)$. Therefore,

$$
\begin{aligned}
\mathbb{P}\left[490 \leq \bar{X} \leq 510\right] & =\Phi\left(\frac{510-500}{\frac{80}{\sqrt{100}}}\right)-\Phi\left(\frac{490-500}{\frac{80}{\sqrt{100}}}\right) \\
& =\Phi(1.25)-\Phi(-1.25)=0.7888
\end{aligned}
$$

To solve (b), we know that $\Phi(x)=0.025$ implies that $x=-1.96$, and $\Phi(x)=0.975$ implies that $x=+1.96$. So

$$
\frac{y-500}{\frac{80}{\sqrt{100}}}=\pm 1.96 \Rightarrow y=484.32 \quad \text { or } \quad y=515.68
$$

Therefore, $\mathbb{P}\left[484.32 \leq \bar{X} \leq 515.68\right]=0.95$.

\subsubsection{Limitation of the Central Limit Theorem}

If we recall the statement of the Central Limit Theorem (Berry-Esseen), we observe that the theorem states only that

$$
\lim _{N \rightarrow \infty} \mathbb{P}\left[\sqrt{N}\left(\frac{\bar{X}-\mu}{\sigma}\right) \leq \varepsilon\right]=\lim _{N \rightarrow \infty} F_{Z_{N}}(\varepsilon)=F_{Z}(\varepsilon)=\Phi(\varepsilon) .
$$



\subsection{CENTRAL LIMIT THEOREM}

Rearranging the terms,

$$
\lim _{N \rightarrow \infty} \mathbb{P}\left[\bar{X} \leq \mu+\frac{\sigma \varepsilon}{\sqrt{N}}\right]=\Phi(\varepsilon) .
$$

This implies that the approximation is good only when the deviation $\varepsilon$ is small.

Let us consider an example to illustrate this idea. Consider a set of i.i.d. exponential random variables $X_{1}, \ldots, X_{N}$, where $X_{n} \sim \operatorname{Exponential}(\lambda)$. Let $S_{N}=X_{1}+\cdots+X_{N}$ be the sum, and let $\bar{X}=S_{N} / N$ be the sample average. Then, according to Chapter 6.4.1, $S_{N}$ is an Erlang distribution $S_{N} \sim \operatorname{Erlang}(N, \lambda)$ with a $\mathrm{PDF}$

$$
f_{S_{N}}(x)=\frac{\lambda^{N}}{(N-1) !} x^{N-1} e^{-\lambda x} .
$$

Practice Exercise 6.10. Let $S_{N} \sim \operatorname{Erlang}(N, \lambda)$ with a $\operatorname{PDF} f_{S_{N}}(x)$. Show that if $Y_{N}=a S_{N}+b$ for any constants $a$ and $b$, then

$$
f_{Y_{N}}(y)=\frac{1}{a} f_{S_{N}}\left(\frac{y-b}{a}\right) .
$$

Solution: This is a simple transformation of random variables:

$$
F_{Y_{N}}(y)=\mathbb{P}[Y \leq y]=\mathbb{P}\left[a S_{N}+b \leq y\right]=\mathbb{P}\left[S_{N} \leq \frac{y-b}{a}\right]=\int_{-\infty}^{\frac{y-b}{a}} f_{S_{N}}(x) d x .
$$

Hence, using the fundamental theorem of calculus,

$$
f_{Y_{N}}(y)=\frac{d}{d y} \int_{-\infty}^{\frac{y-b}{a}} f_{S_{N}}(x) d x=\frac{1}{a} f_{S_{N}}\left(\frac{y-b}{a}\right)
$$

We are interested in knowing the statistics of $\bar{X}$ and comparing it with a Gaussian. To this end, we construct a normalized variable

$$
Z_{N}=\frac{\bar{X}-\mu}{\sigma / \sqrt{N}},
$$

where $\mu=\mathbb{E}\left[X_{n}\right]=\frac{1}{\lambda}$ and $\sigma^{2}=\operatorname{Var}\left[X_{n}\right]=\frac{1}{\lambda^{2}}$. Then

$$
Z_{N}=\frac{S_{N} / N-\mu}{\sigma / \sqrt{N}}=\frac{S_{N}-N \mu}{\sigma \sqrt{N}}=\frac{\lambda}{\sqrt{N}} S_{N}-\sqrt{N}
$$

Using the result of the practice exercise, by mapping $a=\frac{\lambda}{\sqrt{N}}$ and $b=-\sqrt{N}$, it follows that

$$
f_{Z_{N}}(z)=\frac{\sqrt{N}}{\lambda} f_{S_{N}}\left(\frac{z+\sqrt{N}}{\frac{\lambda}{\sqrt{N}}}\right) .
$$

Now we compare $Z_{N}$ with the standard Gaussian $Z \sim \operatorname{Gaussian}(0,1)$. According to the Central Limit Theorem, the standard Gaussian is a good approximation to the normalized

\section{CHAPTER 6. SAMPLE STATISTICS}

sample average $Z_{N}$. To compare the two results, we conduct a numerical experiment. We let $\lambda=1$ and we vary $N$. We plot the $\mathrm{PDF} f_{Z_{N}}(z)$ as a function of $z$, for different $N$ 's, in Figure 6.22 In addition, we plot the PDF $f_{Z}(z)$, which is the standard Gaussian.

The plot in Figure 6.22 shows that while the Central Limit Theorem provides a good approximation, the approximation is only good for values that are close to the mean. For the tails, the Gaussian approximation is not as good.

![](https://cdn.mathpix.com/cropped/2023_02_15_18ba4dd3df9dc0499a5bg-62.jpg?height=517&width=711&top_left_y=525&top_left_x=482)

Figure 6.22: CLT fails at the tails. We note that $X_{1}, \ldots, X_{N}$ are i.i.d. exponential with a parameter $\lambda=1$. We plot the PDFs of the normalized sample average $Z_{N}=\frac{\bar{X}-\mu}{\sigma / \sqrt{N}}$ by varying $N$. We plot the PDF of the standard Gaussian $Z \sim$ Gaussian $(0,1)$ on the same grid. Note that the Gaussian approximation is good for values that are close to the mean. For the tails, the Gaussian approximation is not very accurate.

The limitation of the Central Limit Theorem is attributable to the fact that Gaussian is a second-order approximation. If a random variable has a very large third moment, the second-order approximation may not be sufficient. In this case, we need a much larger $N$ to drive the third moment to a small value and make the Gaussian approximation valid.

\section{When will the Central Limit Theorem fail?}

- The Central Limit Theorem fails when $N$ is small.

- The Central Limit Theorem fails if the third moment is large. As an extreme case, a Cauchy random variable does not have a finite third moment. The Central Limit Theorem is not valid for this case.

- The Central Limit Theorem can only approximate the probability for input values near the mean. It does not approximate the tails, for which we need to use Chernoff's bound.

\section{$6.5$ Summary}

Why do we need to study the sample average? Because it is the summary of the dataset. In machine learning, one of the most frequently asked questions is about the number of training

\subsection{REFERENCES}

samples required to train a model. The answer can be found by analyzing the average number of successes and failures as the number of training samples grows. For example, if we define $f$ as the classifier that takes a data point $\boldsymbol{x}_{n}$ and predicts a label $f\left(\boldsymbol{x}_{n}\right)$, we hope that it will match with the true label $y_{n}$. If we define an error

$$
E_{n}=\left\{\begin{array}{lll}
1, & f\left(\boldsymbol{x}_{n}\right)=y_{n} & \text { correct classification, } \\
0, & f\left(\boldsymbol{x}_{n}\right) \neq y_{n} & \text { incorrect classification }
\end{array}\right.
$$

then $E_{n}$ is a Bernoulli random variable, and the total loss $\mathcal{E}=\frac{1}{N} \sum_{n=1}^{N} E_{n}$ will be the training loss. But what is $\frac{1}{N} \sum_{n=1}^{N} E_{n}$ ? It is exactly the sample average of $E_{n}$. Therefore, by analyzing the sample average $\mathcal{E}$ we will learn something about the generalization capability of our model.

How should we study the sample average? By understanding the law of large numbers and the Central Limit Theorem, as we have seen in this chapter.

- Law of large numbers: $\bar{X}$ converges to the true mean $\mu$ as $N$ grows.

- Central Limit Theorem: The CDF of $\bar{X}$ can be approximated by the CDF of a Gaussian, as $N$ grows.

Performance guarantee? The other topic we discussed in this chapter is the concept of convergence type. There are essentially four types of convergence, ranked in the order of restrictions.

- Deterministic convergence: A sequence of deterministic numbers converges to another deterministic number. For example, the sequence $1, \frac{1}{2}, \frac{1}{3}, \frac{1}{4}, \ldots$ converges to 0 deterministically. There is nothing random about it.

- Almost sure convergence: Randomness exists, and there is a probabilistic convergence. Almost sure convergence means that there is zero probability of failure after a finite number of failures.

- Convergence in probability: The sequence of probability values converges, i.e., the chance of failure is going to zero. However, you can still fail even if your $N$ is large.

- Convergence in distribution: The probability values can be approximated by the CDF of a Gaussian.

\subsection{References}

\section{Moment-Generating and Characteristic Functions}

6-1 Dimitri P. Bertsekas and John N. Tsitsiklis, Introduction to Probability, Athena Scientific, 2nd Edition, 2008. Chapter 4.4.