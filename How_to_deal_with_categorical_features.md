# How to deal with categorical features?

Most of adversarial attack algorithms are designed for image data, which is continuous ([0, 255]). White-box attacks are usually done by adding a small perturbation to the original samples, and they require a distance metric to measure the distance between two samples. For continuous data, we can use Lp-norm to measure the distance.

However, in tabular data, we have to deal with not only numerical data but also categorical data. There are two types of categorical data: nominal data and ordinal data. 

- Nominal data is the type of data used to name variables. We can use one-hot encoding to convert it to numerical data.
- Ordinal data has a scale or order to it. We can use label encoding to convert it to numerical data to preserve the order.

## How recent papers of adversarial attacks on tabular data deal with categorical features?

| Paper | Year | Encoding method | Distance metric |
| --- | --- | --- | --- |
| [Imperceptible Adversarial Attacks on Tabular Data](https://arxiv.org/abs/1911.03274) | 2019 | Drop all categorical features | Lp-norm & Weighted Lp-norm |
| [Not All Datasets Are Born Equal: On Heterogeneous Data and Adversarial Examples](https://doi.org/10.1016/j.knosys.2022.108377) | 2021 | Label encoding |  |
| [FENCE: Feasible Evasion Attacks on Neural Networks in Constrained Environments](https://doi.org/10.1145/3544746) | 2022 | One-hot encoding | L2-norm |
| [Adversarial Attacks for Tabular Data: Application to Fraud Detection and Imbalanced Data](https://arxiv.org/abs/2101.08030) | 2021 | One-hot encoding | L2-norm |
| [Adversarial Robustness for Tabular Data through Cost and Utility Awareness](https://arxiv.org/abs/2208.13058) | 2022 | Discrete continuous features | Use cost |
| [Discretization Inspired Defence Algorithm Against Adversarial Attacks on Tabular Data](https://doi.org/10.1007/978-3-031-05936-0_29) | 2022 | Discrete continuous features |  |
| [PermuteAttack: Counterfactual Explanation of Machine Learning Credit Scorecards](https://arxiv.org/abs/2008.10138) | 2020 | One-hot encoding, but | Lp-norm |


# Related projects

- Alibi Counterfactual: https://docs.seldon.io/projects/alibi/en/latest/methods/CFProto.html#Categorical-Variables