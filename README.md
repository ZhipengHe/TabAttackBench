# TabAttackBench: A Benchmark for Tabular Data Adversarial Attacks

## Introduction

[TODO]

## Background

To find an adversarial example for data point $x$, the problem can be define as follows:

$$ x^{adv} = x + \delta$$

where $\delta$ is the perturbation. The goal is to find $\delta$ such that $x^{adv}$ is misclassified by the model. Then, we introduce the perturbation budget $\epsilon$ to clip the perturbation $\delta$ such that $\|\delta\|_p \leq \epsilon$.

For white-box adversarial attacks, there are two main approaches to find the perturbation $\delta$: bounded attacks and unbounded attacks. For bounded attacks, the perturbation $\delta$ is constrained by the perturbation budget $\epsilon$. For unbounded attacks, the perturbation $\delta$ is not constrained by the perturbation budget $\epsilon$.

### Bounded Attacks

$$max_{\delta} \{J(\theta, x + \delta, y)\}, \text{subject to}\|\delta\|_p \leq \epsilon$$

where $J(\theta, x, y)$ is the loss function, $\theta$ is the model parameters, $x$ is the input data, $y$ is the ground truth label, and $\epsilon$ is the perturbation budget.

1. Fast Gradient Sign Method (FGSM)

    The Fast Gradient Sign Method (FGSM) is a white-box attack. It is a one-step attack that uses the gradient of the loss function to find the perturbation $\delta$.

    $$ \delta = \epsilon \cdot sign(\nabla_x J(\theta, x, y))$$

    where $J(\theta, x, y)$ is the loss function, $\theta$ is the model parameters, $x$ is the input data, $y$ is the ground truth label, and $\epsilon$ is the perturbation budget.

2. Basic Iterative Method (BIM)

    The Basic Iterative Method (BIM) is a white-box attack. It is a multi-step attack that uses the gradient of the loss function to find the perturbation $\delta$.

    $$ x^{adv}_{t+1} = Clip_{x, \epsilon} \{x^{adv}_t + \alpha \cdot sign(\nabla_x J(\theta, x^{adv}_t, y))\}$$

    where $J(\theta, x, y)$ is the loss function, $\theta$ is the model parameters, $x$ is the input data, $y$ is the ground truth label, $\epsilon$ is the perturbation budget, and $\alpha$ is the step size.

3. Projected Gradient Descent (PGD)

    The Projected Gradient Descent (PGD) is a white-box attack. It is a multi-step attack that uses the gradient of the loss function to find the perturbation $\delta$.

    $$ x^{adv}_{t+1} = Clip_{x, \epsilon} \{x^{adv}_t + \alpha \cdot sign(\nabla_x J(\theta, x^{adv}_t, y))\}$$

    where $J(\theta, x, y)$ is the loss function, $\theta$ is the model parameters, $x$ is the input data, $y$ is the ground truth label, $\epsilon$ is the perturbation budget, and $\alpha$ is the step size.

    > Note: The PGD attack is equivalent to the BIM attack when the step size $\alpha$ is equal to the perturbation budget $\epsilon$.

### Unbounded Attacks

Minimizing the distance between the adversarial example $x^{adv}$ and the original data point $x$ is a common approach for unbounded attacks. The distance can be measured by the $L_p$ norm, where $p$ is a positive integer.

$$min_{\delta} \{\|\delta\|_p\}, \text{ subject to } f(x + \delta) \neq f(x)$$

where $f(x)$ is the predicted label of the data point $x$, and $f(x + \delta)$ is the predicted label of the adversarial example $x^{adv}$.


1. Carlini & Wagner (C&W)

    The Carlini & Wagner (C&W) attack is an unbounded attack. For the $L_2$ norm, the C&W attack is defined as follows:

    $$min_{\delta} \{\|\delta\|_2 + c \cdot f(x + \delta)\} $$

    where $f(x + \delta)$ is the confidence of the adversarial example $x^{adv}$, and $c$ is a constant.

    $$ f(x + \delta) = max(max\{Z(x + \delta)_i: i \neq t\} - Z(x + \delta)_t, -\kappa)$$

    where $Z(x + \delta)$ is the logits of the adversarial example $x^{adv}$, $t$ is the ground truth label, $\kappa$ is a constant, and $c$ is a constant.

2. DeepFool

    The DeepFool attack is an unbounded attack. It assumes that the decision boundary of the model is linear and attempts to find the closest decision boundary to the original data point $x$. Given a hyperplane $H = \{x: w^Tx + b = 0\}$, it separates the data points into two classes: $H^+$ and $H^-$. The distance between the data point $x$ and the hyperplane $H$ is defined as follows:

    $$ d(x, H) = -\frac{w^Tx + b}{\|w\|_2} \text{ subject to } f(x) \neq f(x^{adv})$$

    where $f(x)$ is the predicted label of the data point $x$, and $f(x^{adv})$ is the predicted label of the adversarial example $x^{adv}$. 



## Results

The results of the benchmark are available in [result page](./results.html).







## Acknowledgements

We would like to thank these public repositories from which we have borrowed code and ideas:

- **Predictive Models**
    - Implementation of TabTransformer and FTtransformer: https://github.com/lucidrains/tab-transformer-pytorch

- **Attack Methods**
    - Foolbox: https://github.com/bethgelab/foolbox