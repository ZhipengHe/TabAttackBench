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
<!-- | [PermuteAttack: Counterfactual Explanation of Machine Learning Credit Scorecards](https://arxiv.org/abs/2008.10138) | 2020 | One-hot encoding, but | Lp-norm | -->


## Related projects

### [Alibi Counterfactual](https://docs.seldon.io/projects/alibi/en/latest/methods/CFProto.html#Categorical-Variables)

Alibi Counterfactual is a Python library for generating counterfactual explanations for tabular data. It supports both numerical and categorical features. Here is how alibi counterfactual deals with categorical features:

> Our method first computes the pairwise distances between categories of a categorical variable based on either the model predictions (MVDM) or the context provided by the other variables in the dataset (ABDM). For MVDM, we use the difference between the conditional model prediction probabilities of each category. This method is based on the Modified Value Difference Metric (MVDM) by [Cost et al (1993)](https://link.springer.com/article/10.1023/A:1022664626993). ABDM stands for Association-Based Distance Metric, a categorical distance measure introduced by [Le et al (2005)](https://www.sciencedirect.com/science/article/abs/pii/S0167865505001686). ABDM infers context from the presence of other variables in the data and computes a dissimilarity measure based on the Kullback-Leibler divergence. Both methods can also be combined as ABDM-MVDM. We can then apply multidimensional scaling to project the pairwise distances into Euclidean space.

Also see how to calculate mahalanobis distance for categorical features in alibi detect:
https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/mahalanobis.html

#### How to implement `ABDM`, `MVDM` and `ABDM-MVDM`?

1. Use oridinal encoding to convert categorical features to numerical features. If one-hot encoding is used, convert one-hot encoding to ordinal encoding.

```python
# Source: https://github.com/SeldonIO/alibi/blob/54d0c957fb01c7ebba4e2a0d28fcbde52d9c6718/alibi/explainers/cfproto.py#L719-L722

if self.ohe:  # convert OHE to ordinal encoding
    train_data_ord, self.cat_vars_ord = ohe_to_ord(train_data, self.cat_vars)
else:
    train_data_ord, self.cat_vars_ord = train_data, self.cat_vars

```

2. If using `ABDM` or `ABDM-MVDM`, firstly, bin numerical features to compute the pairwise distance matrices

```python
## Source: https://github.com/SeldonIO/alibi/blob/54d0c957fb01c7ebba4e2a0d28fcbde52d9c6718/alibi/explainers/cfproto.py#L724C13-L735
# bin numerical features to compute the pairwise distance matrices
cat_keys = list(self.cat_vars_ord.keys())
n_ord = train_data_ord.shape[1]
numerical_feats = [feat for feat in range(n_ord) if feat not in cat_keys]
# if using ABDM or ABDM-MVDM, and not all features are categorical
if d_type in ['abdm', 'abdm-mvdm'] and len(cat_keys) != n_ord:
    fnames = [str(_) for _ in range(n_ord)]
    disc = Discretizer(train_data_ord, numerical_feats, fnames, percentiles=disc_perc)
    train_data_bin = disc.discretize(train_data_ord)
    cat_vars_bin = {k: len(disc.feature_intervals[k]) for k in range(n_ord) if k not in cat_keys}
# if using MVDM, or all features are categorical
else:
    train_data_bin = train_data_ord
    cat_vars_bin = {}
```

3. 


