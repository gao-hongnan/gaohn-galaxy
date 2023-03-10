# Concept

One key think to note is Logistic Regression variants use **conditional maximum likelihood estimation** while naive bayes uses **joint maximum likelihood estimation**. The difference is the latter maximizes the joint distribution which find parameters that maximize the joint distribution
$\mathbb{P}_{\mathcal{D}}\left[\mathbf{X}, Y ; \boldsymbol{\theta}\right]$, while the former find parameters that maximizes the conditional
distribution $\mathbb{P}_{\mathcal{D}}\left[Y | \mathbf{X} ; \boldsymbol{\theta}\right]$.

## Old Notes

**TO REFACTOR**

- The choice of sigmoid not only satisfies the problems we faced, but it is also mathematically convenient. In reality, we can have other choices other than sigmoid.

1. We have stated that linear regression is not appropriate in the case of a qualitative response. Why not? Suppose that we are trying to predict the medical condition of a patient in the emergency room on the basis of her symptoms. In this simplified example, there are three possible diagnoses: stroke, drug overdose, and epileptic seizure. We could consider encoding these values as a quantitative response variable, $Y$, as follows:
    $$
    \begin{equation}
        Y=\begin{cases}
        1, & \text{if stroke}\\
        2, & \text{if drug overdose}\\
        3, & \text{if epileptic seizure}\\
        \end{cases}
    \end{equation}
    $$

   Using this coding, least squares could be used to fit a linear regression model to predict $Y$ on the basis of a set of predictors $X_{1}$, . . . , $X_{p}$. Unfortunately, this coding implies an ordering on the outcomes, putting drug overdose in between stroke and epileptic seizure, and insisting that the difference between stroke and drug overdose is the same as the difference between drug overdose and epileptic seizure. In practice there is no particular reason that this needs to be the case. For instance, one could choose an
    equally reasonable coding,

    $$
    \begin{equation}
        Y=\begin{cases}
        1, & \text{if epileptic seizure}\\
        2, & \text{if stroke}\\
        3, & \text{if drug overdose}\\
        \end{cases}
    \end{equation}
    $$

   which would imply a totally different relationship among the three conditions. Each of these codings would produce fundamentally different linear models that would ultimately lead to different sets of predictions on test observations.

   If the response variable's values did take on a natural ordering, such as mild, moderate, and severe, and we felt the gap between mild and moderate was similar to the gap between moderate and severe, then a 1, 2, 3 coding would be reasonable. Unfortunately, in general there is no natural way to convert a qualitative response variable with more than two levels into a quantitative response that is ready for linear regression.

   For a binary (two level) qualitative response, the situation is better. For instance, perhaps there are only two possibilities for the patient's medical condition: stroke and drug overdose. We could then potentially use the dummy variable approach to code the response as follows:
    $$
    \begin{equation}
        Y=\begin{cases}
        0, & \text{if stroke}\\
        1, & \text{if drug overdose}\\
        \end{cases}
    \end{equation}
    $$

   We could then fit a linear regression to this binary response, and predict drug overdose if $\hat{Y}>0.5$ and stroke otherwise. In the binary case it is not hard to show that even if we flip the above coding, linear regression will produce the same final predictions.

   For a binary response with a $0-1$ coding as above, regression by least squares does make sense; it can be shown that the $X\hat{\beta}$ obtained using linear regression is in fact an estimate of $\text{Pr}(\text{Drug Overdoes}~|~X)$ in this special case. However, if we use linear regression, some of our estimates might be outside the $[0$, 1$]$ interval , making them hard to interpret as probabilities!

   However, the dummy variable approach cannot be easily extended to accommodate qualitative responses with more than **two levels!!!** For these reasons, it is preferable to use a classification method that is truly suited for qualitative response values, such as the ones presented next.

2. Now we recall back in linear regression, we have independent input/variables $X$ and we seek a response/output variable $Y$. However in logistic regression, our output is in the form of a categorical variable. In our simple tutorial we will only be considering a binary output binary coded as 0 and 1.

   As mentioned in the previous paragraph, we cannot use linear regression to predict a categorical output even if the categorical output is coded as numerical values, especially in such a classification problem with many levels in a categorical variable. As a result, we need to come up with a slightly different **hypothesis** to model our relationship for $X$ and $Y$.

   As with any modelling, there should be a "formula" between the $X$ and the $Y$. However, we have already established it is not easy to obtain an direct equation between $X$ and $Y$. Instead, in logistic regression, we are more interested in having a relation between $X$ and $P(Y = 1 ~|~ X)$. One should immediately be asking, what and why is $P(Y=1~|~X)$? Let me give you an intuition by the following example.

---

3. **Example**: We are trying to predict if a person has malignant tumor or not based on some inputs such as "Tumor Size" and etc. For simplicity sake, we only deal with one variable: The "Tumor Size" $X$.

   Our output $Y$ is basically encoded as a binary class where **Yes it is Malignant Tumor** stands for 1 and **No it is not a Malignant Tumor** stands for 0. Consider the data set above, where the response "$Y$ = Malignant" falls into one of two categories, Yes (1) or No (0). As mentioned in the previous paragraph, rather than modeling this response $Y$ directly with $X$, logistic regression models the probability that $Y$ belongs to a particular category. One can see the from the below figure, the first row says that the tumour size is 1 cm and our data says that the patient's tumour is not malignant (thank god!). But as we said, we want the probability of the tumour being malignant **given** $X = 1$cm. So we write down the probability as shown below (just get this intuition first and don't wonder how are we going to find the probability first!).

    ![1](https://raw.githubusercontent.com/ghnreigns/imagesss/master/logistic-8.jpg)

   For this, in a simple (one variable only) logistic regression model, we can define our output $Y$ as a probability defined as $$P(Y=1~|~ X)$$

   **So to reiterate, instead of modelling our $Y$ directly with $X$, we aim to find a model that can model the probability of $Y$ given $X$.** But why? How does getting a probability help us? Although it should be obvious that $P(Y=1~|~X)$ should fall in between $0$ and $1$ since it is a probability, **but** it does not answer our question of whether you are in class 1 or 0 because ultimately, we are interested in finding out our output value which is either a 1 or a 0.

   I DO NOT CARE if you told me you found that  $P(Y=1~|~X =1.1 \text{ cm}) = 0.2349538$ which is neither 0 or 1? So this is where **classification threshold** comes in. You need to pre-define a threshold (default is usually 0.5). As a result, if we use a classification threshold of 0.5, then we will predict a $Y = \text{Yes } (1)$ for any $P(Y=1|~X) > 0.5$. To write it more compactly, we define the following indicator function. (For more experienced peeps, there is connection with AUC,ROC curves here as well!)

    $$
    \begin{equation}
        Y=\begin{cases}
        1, & \text{if $P(Y=1~|~X) \geq 0.5$}\\
        0, & \text{if $P(Y=1~|~X) < 0.5$}\\
        \end{cases}
    \end{equation}
    $$

   As a result if your tumour size is 1.1 cm, then the probability of your tumour is malignant is $0.2349538$, which is less than 0.5 and we predict your as a No (not malignant). However, the threshold is there for a reason, usually, in medical and healthcare industry, we tend to be more conservative with our predictions as we have 0 tolerance for False Negatives. We rather give you a false alarm than to classify you as No Malignancy when in fact you are already at the last stage of your life. So we can tune and change our threshold to something like $P(Y=1~|~X) >0.1$ and in this case, $0.2349538$ will be in the Yes class.

   I know we are going off the tracks, but I hope I have provided you with some intuition on how modelling the $P(Y=1~|~X)$ as a function of $X$ makes sense here.

---

4. **Hypothesizing and Modelling the Logistic Function:** For simplicity we call our function $P(Y=1~|~ X)$ as the function $p(X)$ and we seek to find a relationship between $p(X)$ and $X$. Although we have gone through a lot of ideas just now, it would be meaningless if we cannot find a suitable function (equation) to model $p(X)$ and $X$.

   **Hypothesis 1: The Linear Hypothesis**

   Hmm, so we got quite some success hypothesizing linear regression models with linear functions, can we try that too on $p(X)$ and $X$? Consider that we "guess/hypothesize" that $p(X)$ have a **linear relationship** with $X$ as follows:

    $$
    p(X) = \beta_0+\beta_1 X
    $$

   However the problem with this modelling is that for very large Tumour sizes $X$, say $X=10 \text{ cm}$, then our $p(X)$ may take values greater than $1$. And for extremely small Tumour sizes $X$, say those very small benign lumps, which may be $X=0.05 \text{ cm}$ in size, then $p(X)$ may take negative values. In any case, no matter how likely or unlikely one is to find his/her tumour to be malignant, how big or small the tumour size is, our $p(X)$ should only output values between $0$ and $1$ because $p(X)$ is a probability. Hence our linear model may be accurate to a certain extent, but not sensible.

   **Hypothesis 2**

   Instead of the linear hypothesis, we come up with another one, recall that we learnt that probability and odds have similar definition. And recall that

   $$
   \text{odds} = \dfrac{P(Y=1~|~X)}{1-P(Y=1~|~X)}
   $$

   So why not model the odds against $X$? If we can successfully do that, then we can easily get the probability $P(Y=1~|~X)$ since odds and probability are in a if and only if relationship. So let us try:

   $$
   \text{odds} = \dfrac{p(X)}{1-p(X)} = \beta_0 + \beta_1X
   $$

   But ALAS! We soon realise that the odds can only take on values from $0$ to $\infty$, but the problem still exists for the $\beta_0 + \beta_1X$ since some $X$ values can output negative values.

   But we are close, and if one has **some** mathematical backgrounds, then we know that if we take the log or ln of $\text{odds}$ then we can have the desired results.

   **Hypothesis 3: The Chosen one**

   If we finally consider the modelling of the logarithm of the odds, against the variable $X$, where we still assume a linear relationship, then we may be good to go because the logarithm of the odds gives a range of $-\infty$ to $\infty$ and matches well with $\beta_0+\beta_1X$.

   With this we have achieved a regression model, with the output of the model being the logarithm or ln of the odds. i.e: the modelled equation is as follows:

    $$
    \ln\left(\dfrac{p(X)}{1-p(X)}\right) = \beta_0+\beta_1X
    $$

   The main reason we reach this step is because both sides of the equation can take in the same range, and thus makes more mathematical sense now. We have yet to estimate or found what the coefficients $\beta_0, \beta_1$ are. This is just a logical and sound hypothesis.

   **Recovering the Logistic Function from log odds**

   So in the previous paragraph we have settled on a hypothesis that there is a **linear relationship** between the predictor variable $X$ and the **log-odds** of the event that $Y=1$. However, do not forget what our original aim is, we modelled log odds against $X$ simply because the relationship can be mathematically justified, we ultimately want to find the probability of $Y=1$ given $X$. And that is easy, by some reverse engineering, once $\beta_0, \beta_1$ are fixed, we do some manipulation:


$$
\ln\left(\dfrac{p(X)}{1-p(X)}\right) = \beta_0+\beta_1X \iff  \dfrac{p(X)}{1-p(X)} = \exp{(\beta_0+\beta_1X)}   \iff p(X)  = \dfrac{\exp{(\beta_0+\beta_1X)}}{\exp{(\beta_0+\beta_1X)}+1} \iff p(X) = \dfrac{1}{1+\exp^{-1}{(\beta_0+\beta_1X)}}
$$


   **Given the log odd mode (logit model actually), we can recover the probability of $Y=1$ given $X$ for each $X$.**

---

5.  **Important - The workflow process of Logistic Regression**

    - Given an indepedent variable $X$, we aim to predict a binary dependent variable $Y$.

    - It is not easy to model a relationship between $X$ and $Y$ directly, instead, we find the probability of $Y = 1$ given $X$ instead. Imagine we are in the shoes of the famous statistician DR Cox in the year 1958, we are building logistic regression from scratch, and we tried to hypothesize that the probability $p(X) = P(Y=1~|~X)$ can be modelled the same way as **linear regression?** But soon realised that modelling $p(X) = \beta_0+\beta_1X$ is not good since its range gives values out of $[0,1]$. In order to overcome this we can make a transformation and fit the sigmoid/logistic function which forces the output $p(X)$ to be in $[0,1]$.

    - Since the transformation may not be intuitive, I have made a simple explanation above, and showed steps on how to model $P(X)$ as a sigmoid function.

        Sigmoid in logistic regression:

    $$
    p(X) = \dfrac{1}{1+\exp^{-1}{(\beta_0+\beta_1X)}}
    $$

    - So, we have effectively build a model, and in fact it is a probabilistic model behaving as a bernoulli distribution. To recover the probability $p(X)$, we have to estimate the coefficients (parameters) in

  $$
  p(X) = \dfrac{1}{1+\exp^{-1}{(\beta_0+\beta_1X)}}
  $$

  and we use a method called **Maximum Likelihood (I will do a part 2 on this as it is also a big topic)**.

    - Once we recover the coefficients $\beta_0, \beta_1$, we can simply plug in the coefficients and the respective values of $X$ to get $p(X)$.

    - Once we get the $p(X)$, we can define a indicator function as our classification threshold (mentioned earlier) and subsequently, get all the values of $Y$.


First you need to be very clear about what a **probability distribution is.** Consider that we have 10 students and we model their marks where the full marks of the test is 16/16. Define a random variable $X$ where $X$ represents the marks of each student. Assume further that this **random variable** $X$ is following a **normal distribution** with $\mu = 11$ and $\sigma = 3$, can we find the probability distribution for the marks of the whole cohort (10 students)? Yes we can, because we have the parameters. If you do not know what is the meaning of parameters, please go revise on it, it is very important for you to understand that the **parameter** decides the probability distribution of any model.

Recall the general formula for the pdf of the normal distribution is $$f(X = x) = \dfrac{e^{-(x-\mu)^2}/(2\sigma^2)}{\sigma \sqrt{2\pi}}$$
and in normal distribution once we have the mean and standard deviation of the dataset, we can recover the whole pdf of the model, hence the mean and standard deviation are our parameters.  So let us say we want to find $P(11 < X < 13~|~\mu = 11, \sigma = 3)$, we can easily find it to be around $0.31 = 31\%$, we can basically find any probabilities **as long as we are given the parameters**. So, **we must have the correct mindset that,** probability density functions (or pmf alike) are legitimate functions that takes in any $X = x$ and outputs the probability of this $x$ **happening**. (Of course in continuous distribution we are usually only interested in the range of $x$, but for the purpose of intuition, we do not need to be so pedantic).

**Likelihood Function**

However, in the real world setting, more often than not, we have the data $X$, like we have conveniently the scores of all the 10 students above, which **could be a random sample taken** from the whole school's population. Now we are tasked to find the probability distribution of the whole population (say 10,000 students), and we would have calculate it ever so easily **if we knew what the parameters were!?** Unfortunately we do not have the true parameters.

Our main motivation now is to find the **parameter**, because without it, we cannot complete the task of finding the distribution of the population. We can never know the real/true parameter $\theta = (\mu, \sigma)$, but we can obtain a good estimator of it, by making use of the data that we do have! In this scenario we were given 10 data points (in real life it is usually much more), say the 10 data points are $$\mathbf{X} = [3,9,4,10,12,16,5,11,9,9]$$

So we do a sleight of hand using our original **probability density function**, $P(X = x~|~ \theta)$. Instead of being a function of $X = x$ where $\theta$ is known, we instead let $X = x$ be fixed, and let $\theta$ be the variable now. The idea is that this function now is **NO LONGER a function of** $X=x$, and is instead a function of $\theta$, where it takes in all possible values of $\theta$, and outputs a value called the \textbf{likelihood value.} So now, in a less informal way, our new function looks like $$P(\mathbf{X} = [3,9,4,10,12,16,5,11,9,9]~|~ \theta)$$ and it means **what is the probability of OBSERVING these data points, given different values of theta.** One needs to plot the graph of likelihood out to get a better idea (Wikipedia).

So imagine our function (plot likelihood value vs parameter) has a local/global maximum, and that maximum is what we are finding ultimately. Because it is reasonable for us to believe that, the **parameter** that gives us the maximum value of $P(\mathbf{X} = [3,9,4,10,12,16,5,11,9,9]~|~ \theta)$ will suggest that **given these 10 data points**, this $\theta$ that we just found, gives us the highest **likelihood/probability** that these 10 points are actually observed.

We formally define this function to be $$\mathcal{L}(\theta~|~ X = x) = P(X = x~|~\theta) $$

I cannot emphasize enough that even those the likelihood function $\mathcal{L}$ and the probability function $P$ have the exact same form, they are fundamentally different in which one is a function of the parameter $\theta$, and the other is a function of the data points $X = x$.

**Maximizing the likelihood function**

Many of us may not have rigourous mathematical background, and hence when I said we are trying to find the value of $\theta$ that maximizes $\mathcal{L}$, we might be stunned, indeed, the most intuitive way is to try out every possible theta and see which theta gives us the highest value; or if you can try differentiating $\mathcal{L}$ and set the derivative to $0$ to solve for $\theta$. In any case, our intuition is to maximize this function. We will illustrate the idea of maximum likelihood with a simple example below.

**Example**

Inspired and credits to [Jonny Brooks-Bartlett](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1) as the example is almost similar.

- Let's say we have a population of 1000 students and a random sample of 3 university students are taken with their respective marks for a Machine learning test. The data are distributed below in python (generate some data of 3 students marks).

- First, we always have to come up with a hypothesized model first, and in this case, based on these 3 sample data points, we make a simple plot and since the plot looks somewhat normal, we hypothesize that our model should be of a normal distribution (do be reminded this is just for illustration, 3 points may be too little for us to hypothesize a model in many scenarios).

- Our aim is to come up with a normal distribution for this 1000 students marks. In order to do that, we need to find the parameters and recall that in normal distribution, our parameters are the mean $\mu$ and the standard deviation  $\sigma$. So we have to find the best parameters that best describe our observed 3 sample points.

- Note that from our previous section, we know that we just need to maximize the likelihood function given by $$\mathcal{L}\left(\theta = (\mu, \sigma)~|~ X = [9,9.5,11]\right) = P\left(X = [9,9.5,11]~;~ \theta = (\mu, \sigma)\right)$$

- Here the author made a very good point that we shall use ; instead of | to indicate probability function. However some people like to use | as probability function can be described in a conditional way.

- And since our assumption that this 3 data points are independent of each other, our joint pdf equation can be simplified to


$$P\left(X = [9,9.5,11]~;~ \theta = (\mu, \sigma)\right) = P(9;\theta = [\mu, \sigma]) \cdot P(9.5;\theta = [\mu, \sigma]) \cdot  P(11;\theta = [\mu, \sigma]) = \dfrac{1}{\sigma\sqrt{2\pi}}\exp\left(-\dfrac{(9-\mu)^2}{2\sigma^2}\right) \cdot \dfrac{1}{\sigma\sqrt{2\pi}}\exp\left(-\dfrac{(9.5-\mu)^2}{2\sigma^2}\right) \cdot \dfrac{1}{\sigma\sqrt{2\pi}}\exp\left(-\dfrac{(11-\mu)^2}{2\sigma^2}\right)~~~~~~~~(2.1)$$

Equation 2.1 above is the joint probability of observing these 3 points and is a function of the parameters, so we suffice to find the parameters that maximize this function. So the intuition is to differentiate equation 2.1 and set it to 0, and solve for the parameter $\theta$.

- However, the expression in 2.1 is very difficult to differentiate, and that is why it is very common to see we take log/ln on both sides of the expression

$\ln\mathcal{L} = \ln\left[\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(9-\mu)^2}{2\sigma^2}\right) \cdot \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(9.5-\mu)^2}{2\sigma^2}\right) \cdot \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(11-\mu)^2}{2\sigma^2}\right) \right]$

and instead maximize the new log-likelihood function instead. But some may be confused, when we solve the \textbf{parameters} for the above equation, will our parameter still be the one that actually maximizes our equation 2.1? Yes, it does, the proof can be found here (for those who are curious). [proof](https://math.stackexchange.com/questions/647835/why-does-finding-the-x-that-maximizes-lnfx-is-the-same-as-finding-the#_=_)

- We simplify the above $\ln\mathcal{L}$ function to get $$\ln\mathcal{L} = -3\ln(\sigma)-1.5\ln(2\pi) - \frac{1}{2\sigma^2}\left[(9-\mu)^2+(9.5-\mu)^2+(11-\mu)^2\right]~~~(2.2)$$

- We take the partial derivatives with respect to $\mu$ and $\sigma$ respectively and set to $0$ to solve equation 2.2, thereby getting our $\theta$.

$$\dfrac{\partial \ln(\mathcal{L})}{\partial \mu} = \dfrac{1}{\sigma^2}[9+9.5+11-3\mu] = 0 \implies \mu = 9.833$$

$$\dfrac{\partial \ln(\mathcal{L})}{\partial \sigma} = -\dfrac{3}{\sigma} + 4\sigma^{-3}\left[(9-\mu)^2+(9.5-\mu)^2+(11-\mu)^2\right] =0 \implies \sigma = 1.7$$

- So now we have found the estimate of the real parameter of the population, and we can therefore infer that our populations' probability density function.

- However, we are not really done yet, this example was easy and therefore can be calculated with hand. In reality, there is no closed form solution expression for the coefficient values that maximize the likelihood function, so that an iterative process must be used instead; for example Newton's method. This process begins with a tentative solution, revises it slightly to see if it can be improved, and repeats this revision until no more improvement is made, at which point the process is said to have converged.(Wikipedia)

**It is important to realise that if convergence for the model is not reached, the coefficients/parameters you obtained may not be meaningful. Therefore, certain assumptions need to be fulfilled for logistic regression.**

**Assumptions of logistic regression and Regularization will be mentioned in part III, which is also the final part of the Logistic Regression Trilogy**.

## Logistic Regression