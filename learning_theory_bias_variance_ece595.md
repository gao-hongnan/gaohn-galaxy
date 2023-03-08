
\subsection{Bias and Variance Analysis}

The bias-variance analysis is an alternative way of analyzing the out-sample error. Instead of defining the out-sample error as the probability $\mathcal{R}_{\mathcal{D}}(h_S)=\mathbb{P}[h_S(\mathbf{x}) \neq f(\mathbf{x})]$, bias-variance analysis defines the out-sample error using the squared error:

$$
\mathcal{R}_{\mathcal{D}}\left(h_S^{(\mathcal{S})}\right)=\mathbb{E}_{\mathbf{x}}\left[\left(h_S^{(\mathcal{S})}(\mathbf{x})-f(\mathbf{x})\right)^{2}\right]
$$

One thing to note is that we make the dependency on the training dataset $\mathcal{S}$ explicit. The reason will be clear later. What this means is that if we use a different training set $\mathcal{S}$, we will get a different $\mathcal{R}_{\mathcal{D}}\left(h_S^{(\mathcal{S})}\right)$. This will give us many $\mathcal{R}_{\mathcal{D}}\left(h_S^{(\mathcal{S})}\right)$, depending on how the training sets $\mathcal{S}$ 's are generated. To account for all the possible $\mathcal{S}$ 's, we can compute the expectation and define the expected out-sample error:

$$
\begin{aligned}
\mathbb{E}_{\mathcal{S}}\left[\mathbb{E}_{\text {out }}\left(h_S^{(\mathcal{S})}\right)\right] & =\mathbb{E}_{\mathcal{S}}\left[\mathbb{E}_{\mathbf{x}}\left[\left(h_S^{(\mathcal{S})}(\mathbf{x})-f(\mathbf{x})\right)^{2}\right]\right] \\
& =\mathbb{E}_{\mathbf{x}}\left[\mathbb{E}_{\mathcal{S}}\left[\left(h_S^{(\mathcal{S})}(\mathbf{x})-f(\mathbf{x})\right)^{2}\right]\right] \\
& =\mathbb{E}_{\mathbf{x}}[\mathbb{E}_{\mathcal{S}}\left[h_S^{(\mathcal{S})}(\mathbf{x})^{2}\right]-2 \underbrace{\mathbb{E}_{\mathcal{S}}\left[h_S^{(\mathcal{S})}(\mathbf{x})\right]}_{\bar{h_S}(\mathbf{x})} f(\mathbf{x})+f(\mathbf{x})^{2}]
\end{aligned}
$$

 Here, we define $\bar{h_S}(\mathbf{x})=\mathbb{E}_{\mathcal{S}}\left[h_S^{(\mathcal{S})}(\mathbf{x})\right]$, which can be considered as the asymptotic limit of the estimate $\bar{h_S}(\mathbf{x}) \approx \frac{1}{K} \sum_{k=1}^{K} g_{k}(\mathbf{x})$ as $K \rightarrow \infty$. The hypotheses $g_{1}, \ldots, g_{K}$ are the final hypothesis returned by using the training sets $\mathcal{S}_{1}, \ldots, \mathcal{S}_{K}$. Therefore, for any fixed $\mathbf{x}, g_{k}(\mathbf{x})$ is a random variable over the training set $\mathcal{S}_{k}$. However, one should be careful that even if $g_{1}, \ldots, g_{K}$ are inside the hypothesis set, the mean $\bar{h_S}$ is not necessarily inside too.

Let us do some additional calculation:

$$
\begin{aligned}
& \mathbb{E}_{\mathcal{S}}\left[\mathbb{E}_{\text {out }}\left(h_S^{(\mathcal{S})}\right)\right] \\
= & \mathbb{E}_{\mathbf{x}}\left[\mathbb{E}_{\mathcal{S}}\left[h_S^{(\mathcal{S})}(\mathbf{x})^{2}\right]-2 \mathbb{E}_{\mathcal{S}}\left[h_S^{(\mathcal{S})}(\mathbf{x})\right] f(\mathbf{x})+f(\mathbf{x})^{2}\right] \\
= & \mathbb{E}_{\mathbf{x}}[\underbrace{\left[\mathbb{E}_{\mathcal{S}}\left[h_S^{(\mathcal{S})}(\mathbf{x})^{2}\right]-\bar{h_S}(\mathbf{x})^{2}\right.}_{\mathbb{E}_{\mathcal{S}}\left[\left(h_S^{(\mathcal{S})}(\mathbf{x})-\bar{h_S}(\mathbf{x})\right)^{2}\right]}+\underbrace{\bar{h_S}(\mathbf{x})^{2}-2 \mathbb{E}_{\mathcal{S}}\left[h_S^{(\mathcal{S})}(\mathbf{x})\right] f(\mathbf{x})+f(\mathbf{x})^{2}}_{(\bar{h_S}(\mathbf{x})-f(x))^{2}}] .
\end{aligned}
$$

Based on this decomposition, we can define two terms:

$$
\begin{aligned}
\operatorname{bias}(\mathbf{x}) & \stackrel{\text { def }}{=}(\bar{h_S}(\mathbf{x})-f(\mathbf{x}))^{2}, \\
\operatorname{var}(\mathbf{x}) & \stackrel{\text { def }}{=} \mathbb{E}_{\mathcal{S}}\left[\left(h_S^{(\mathcal{S})}(\mathbf{x})-\bar{h_S}(\mathbf{x})\right)^{2}\right] .
\end{aligned}
$$

The first term is called the bias, as it measures the deviation between the average function $\bar{h_S}(\mathbf{x})$ and the target function $f(\mathbf{x})$. Thus, regardless of how we pick the particular training set, there is an intrinsic gap between the what we would expect $(\bar{h_S}(\mathbf{x}))$ and the ideal target $f(\mathbf{x})$. The second term is called the variance. It measures the variance of the random variable $h_S^{(\mathcal{S})}(\mathbf{x})$ with respect to its mean $\bar{h_S}(\mathbf{x})$. Using the bias and variance decomposition, we can show that

$$
\begin{aligned}
\mathbb{E}_{\mathcal{S}}\left[\mathbb{E}_{\text {out }}\left(h_S^{(\mathcal{S})}\right)\right] & =\mathbb{E}_{\mathbf{x}}[\operatorname{bias}(\mathbf{x})+\operatorname{var}(\mathbf{x})] \\
& =\text { bias }+\text { var },
\end{aligned}
$$

where bias $=\mathbb{E}_{\mathbf{x}}[\operatorname{bias}(\mathbf{x})]$ is the average bias over the distribution $p(\mathbf{x})$, and $\operatorname{var}=\mathbb{E}_{\mathbf{x}}[\operatorname{var}(\mathbf{x})]$ is the average variance over $p(\mathbf{x})$.

What can we say about the bias-variance decomposition when analyzing the model complexity? We can consider two extreme cases. In the first case, we have a very simple model and so $\mathcal{H}$ is small. Since there are not many choices of the hypothesis, the deviation between the target $f$ and the average of these hypotheses $\bar{h_S}$ is large. Thus, the bias is large. On the other hand, the variance is limited because we only have very few hypotheses in $\mathcal{H}$.

The second case is when we have a complex model. By selecting different training sets D's, we will be able to select hypothesis functions $g_{1}, \ldots, g_{K}$ that agree with $f$. In this case, the deviation between the target $f$ and the average of these hypotheses $\bar{h_S}$ is very small. The bias is thus bias $\approx 0$. The variance, however, is large because there are many training sets under consideration.


![](https://cdn.mathpix.com/cropped/2023_02_26_b748253e20a8cfd225a0g-24.jpg?height=264&width=836&top_left_y=338&top_left_x=642)

Figure 4.7: [Left] Large bias but small variance. [Right] Small bias but large variance.

\section{Demonstration}

Consider a target function $f(x)=\sin (\pi x)$ and a dataset of size $N=2$. We sample uniformly in the interval $[-1,1]$ to generate a data set containing two data points $\left(x_{1}, y_{1}\right)$ and $\left(x_{2}, y_{2}\right)$. We want to use these two data points to determine which of the following two models are better:

- $\mathcal{M}_{0}=$ Set of all lines of the form $h(x)=b$

- $\mathcal{M}_{1}=$ Set of all lines of the form $h(x)=a x+b$.

Figure ?? illustrates an example of how the models would yield the lines. Given two data points, $\mathcal{M}_{0}$ seeks a horizontal line $h(x)=b$ that matches the two data points. This line must be the one that passes through the mid-point of the two data points. The model $\mathcal{M}_{1}$ is allowed to find an arbitrary straight line that matches the two data points. Since there are only two data points, the best straight must be the one that passes through both of them. More specifically, the line returned by $\mathcal{M}_{0}$ is

$$
h(x)=\frac{y_{1}+y_{2}}{2}
$$

and the line returned by $\mathcal{M}_{1}$ is

$$
h(x)=\left(\frac{y_{2}-y_{1}}{x_{2}-x_{1}}\right) x+\left(y_{1} x_{2}-y_{2} x_{1}\right) .
$$

As we change $\left(x_{1}, y_{1}\right)$ and $\left(x_{2}, y_{2}\right)$, we will obtain different straight lines.
![](https://cdn.mathpix.com/cropped/2023_02_26_b748253e20a8cfd225a0g-24.jpg?height=334&width=1292&top_left_y=2023&top_left_x=402)

Figure 4.8: [Left] Fitting two data points using $\mathcal{M}_{0}$. [Right] Fitting two data points using $\mathcal{M}_{1}$.

 If we keep drawing two random samples from the sine function, we will eventually get a set of straight lines for both cases. However, since $\mathcal{M}_{0}$ restricts ourselves to horizontal lines, the set of straight lines are all horizontal. In contrast, the set of straight lines for $\mathcal{M}_{1}$ contains lines of different slopes and $y$-intercepts.
![](https://cdn.mathpix.com/cropped/2023_02_26_b748253e20a8cfd225a0g-25.jpg?height=348&width=1308&top_left_y=612&top_left_x=390)

Figure 4.9: [Left] Possible lines generated by $\mathcal{M}_{0}$. [Right] Possible lines generated by $\mathcal{M}_{1}$.

As we increase the number of experiments, the set of straight lines will form a distribution of the model. Since now we have a distribution, we can determine is mean, which is a function, as $\bar{h_S}$. Similarly, we can determine the variance of the function $\operatorname{var}(x)$. For example, in Figure ?? we draw the possible lines that are within one standard deviation from the mean function, i.e., $\bar{h_S} \pm \sqrt{\operatorname{var}(x)}$.
![](https://cdn.mathpix.com/cropped/2023_02_26_b748253e20a8cfd225a0g-25.jpg?height=390&width=1292&top_left_y=1369&top_left_x=402)

Figure 4.10: Average hypothesis function $\bar{h_S}(x)$ and the variance $\operatorname{var}(x)$.

So which model is better in terms of bias-variance? If we compute the bias and variance, we can show that

$$
\begin{array}{ll}
\operatorname{bias}_{\mathcal{M}_{0}}=0.5, & \operatorname{bias}_{\mathcal{M}_{1}}=0.21 \\
\operatorname{var}_{\mathcal{M}_{0}}=0.25, & \operatorname{var}_{\mathcal{M}_{1}}=1.69
\end{array}
$$

Therefore, as far as generalization is concerned, a simple model using a horizontal line is actually more preferred in the bias-variance sense. This is counter-intuitive because how can a horizontal line with only one degree of freedom be better than a line with two degrees of freedom when approximating the sine function? However, the objective here is not to use a line to approximate a sine function because we are not supposed to observe the entire sine function. Remember, we are only allowed to see two data points and our goal is to

 construct a line based on these two data points. The approximation error in the usual sense is captured by the bias, as $\bar{h_S}$ is the best possible line within the class. The generalization, however, should also take into account of the variance. While $\mathcal{M}_{1}$ has a lower bias, its variance is actually much larger than that of $\mathcal{M}_{0}$. The implication is that while on average $\mathcal{M}_{1}$ performs well, chances are we pick a bad line in $\mathcal{M}_{1}$ that end up causing very undesirable out-sample performance.

One thing to pay attention to is that the above analysis is based on $N=2$ data points. If we increase the number of data points, the variance of $\mathcal{M}_{1}$ will drop. As $N \rightarrow \infty$, the variance of both $\mathcal{M}_{0}$ and $\mathcal{M}_{1}$ will eventually drop to zero and so only the bias term matters. Therefore, if we have infinitely many training data, a complex model will of course provide a better generalization.

\section{Bias-Variance and VC Dimension}

There is a subtle but important difference between the bias-variance analysis and the $\mathrm{VC}$ analysis. Bias-variance depends on the learning algorithm $\mathcal{A}$ whereas the VC analysis is independent of $\mathcal{A}$. With the same hypothesis set $\mathcal{H}, \mathrm{VC}$ will always return the same generalization bound. This is a uniform performance guarantee over all possible choices of dataset $\mathcal{S}$. For bias-variance, the same $\mathcal{H}$ can lead to different $h_S^{(\mathcal{S})}$, depending of which $\mathcal{S}$ is being used. This is reflected in the bias and variance term $\mathcal{R}_{\mathcal{D}}\left(h_S^{(\mathcal{S})}\right)$. Of course, the overall biasvariance is independent of $\mathcal{S}$ because we take expectation $\mathbb{E}_{\mathcal{S}}\left[\mathcal{R}_{\mathcal{D}}\left(h_S^{(\mathcal{S})}\right)\right]$. VC analysis does not have this issue.

In practice, bias and variance cannot be computed because we never have the target function. (If we know the target function there is nothing to learn!) Therefore, bias-variance can only be served as a conceptual tool to guide the design of a learning algorithm. For example, one can try to reduce the bias but maintaining the variance (e.h_S., via regularization and prior), or reduce the variance but maintaining the bias.

\section{Learning Curve}

Both bias-variance and VC analysis provide a trade-off between model complexity and sample complexity. Figure ?? shows a typical scenario. Suppose that we have learned an final hypothesis $h_S^{(\mathcal{S})}$ using dataset $\mathcal{S}$ of size $N$. This final hypothesis will give us an in-sample error $\hat{\mathcal{R}}_{\mathcal{S}}\left(h_S^{(\mathcal{S})}\right)$ and out-sample error $\mathcal{R}_{\mathcal{D}}\left(h_S^{(\mathcal{S})}\right)$. These two errors are functions of the dataset $\mathcal{S}$. If we take the expectation over $\mathcal{S}$, we will obtain the expected error $\mathbb{E}_{\mathcal{S}}\left[\hat{\mathcal{R}}_{\mathcal{S}}\left(h_S^{(\mathcal{S})}\right)\right]$ and $\mathbb{E}_{\mathcal{S}}\left[\mathcal{R}_{\mathcal{D}}\left(h_S^{(\mathcal{S})}\right)\right]$. These expected error will give us two curves, as shown in Figure ??.

If we have a simple model, the in-sample error $\mathbb{E}_{\mathcal{S}}\left[E_{\mathrm{in}}\left(h_S^{(\mathcal{S})}\right)\right]$ is a good approximate of the out-sample error $\mathbb{E}_{\mathcal{S}}\left[\mathcal{R}_{\mathcal{D}}\left(h_S^{(\mathcal{S})}\right)\right]$. This implies a small gap between the two. However, the overall expected error could still be large because our model is simple. This is reflected in the high off-set in the learning curve.

 If we have a complex model. the in-sample error $\mathbb{E}_{\mathcal{S}}\left[\hat{\mathcal{R}}_{\mathcal{S}}\left(h_S^{(\mathcal{S})}\right)\right]$ would be small because we are able to fit the training data. However, the out-sample error is large $\mathbb{E}_{\mathcal{S}}\left[\mathcal{R}_{\mathcal{D}}\left(h_S^{(\mathcal{S})}\right)\right]$ because the generalization using a complex model is difficult. The two curves will eventually meet as $N$ grows, since the variance of the out-sample will drop. The convergence rate is slower than a simple model, because it take many more samples for a complex model to generalize well.
![](https://cdn.mathpix.com/cropped/2023_02_26_b748253e20a8cfd225a0g-27.jpg?height=496&width=1300&top_left_y=763&top_left_x=390)

Figure 4.11: Learning curves of a simple and a complex model.

The VC analysis and the bias-variance analysis provide two different view of decomposing the error. VC analysis decompose $\mathcal{R}_{\mathcal{D}}$ as the in-sample error $\hat{\mathcal{R}}_{\mathcal{S}}(h_S)$ and the generalization error $\epsilon$. This $\epsilon$ is the gap between $\hat{\mathcal{R}}_{\mathcal{S}}$ and $\mathcal{R}_{\mathcal{D}}$. The bias-variance analysis decompose $\mathcal{R}_{\mathcal{D}}$ as bias and variance. The bias is the residue caused by the average hypothesis $\bar{h_S}$. The bias is a fixed quantity and does not change over $N$. The gap between $\mathcal{R}_{\mathcal{D}}$ and the bias is the variance. The variance drops as $N$ increases.
![](https://cdn.mathpix.com/cropped/2023_02_26_b748253e20a8cfd225a0g-27.jpg?height=472&width=1302&top_left_y=1810&top_left_x=390)

Figure 4.12: [Left] VC analysis. [Right] Bias-variance analysis



\subsection{Validation}

\subsection{Practical Considerations}

