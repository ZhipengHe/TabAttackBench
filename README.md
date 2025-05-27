# TabAttackBench: A Benchmark for Tabular Data Adversarial Attacks

## Abstract

Adversarial attacks pose a significant threat to machine learning models by inducing incorrect predictions through imperceptible perturbations to input data. While these attacks have been extensively studied in unstructured data like images, their application to tabular data presents new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ significantly from those in image data. To address these differences, it is crucial to consider imperceptibility as a key criterion specific to tabular data. Most current research focuses primarily on achieving effective adversarial attacks, often overlooking the importance of maintaining imperceptibility. To address this gap, we propose a new benchmark for adversarial attacks on tabular data that evaluates both effectiveness and imperceptibility. In this study, we assess the effectiveness and imperceptibility of five adversarial attacks across four models using eleven tabular datasets, including both mixed and numerical-only datasets. Our analysis explores how these factors interact and influence the overall performance of the attacks. We also compare the results across different dataset types to understand the broader implications of these findings. The findings from this benchmark provide valuable insights for improving the design of adversarial attack algorithms, thereby advancing the field of adversarial machine learning on tabular data.

## Overview

TabAttackBench is a benchmark suite for evaluating adversarial attacks on tabular data. It focuses on both the effectiveness and imperceptibility of attacks, providing a comprehensive comparison across multiple models and datasets.

## Project Structure

- **attacks/**: Implementations and configuration for adversarial attack methods.
- **models/**: Contains model definitions, training scripts, and assets for predictive models (including PyTorch models).
- **utils/**: Utility functions and helpers used throughout the project.
- **data/**: Dataset loading, configuration, and management scripts. Includes subfolders for raw datasets and assets.
- **datapoints/**: Stores generated adversarial examples and perturbed data for each attack/dataset/model combination.
- **results/**: Aggregated results, figures, and summary CSVs from experiments. Includes pickled results for each model/attack/dataset.
- **configs/**: Configuration files for attacks, models, and experiment settings.
- **result_evaluation.ipynb**: Jupyter notebook for evaluating and visualizing benchmark results.
- **generate_adversarial_examples.ipynb**: Notebook for generating adversarial examples using different attacks.
- **model_training.ipynb**: Notebook for training predictive models on tabular datasets.
- **model_hyperparameter_search.ipynb**: Notebook for hyperparameter tuning of models.
- **dataset_profiling.ipynb**: Notebook for profiling and analyzing dataset characteristics.
- **best_epsilons.csv**: Stores the best perturbation budgets ($\epsilon$) found for each experiment.
- **results.html**: Main results page for browsing benchmark outcomes.

## Dataset Profiles

Below is a summary of the 11 datasets used in the benchmark, including the total number of instances, splits, and feature counts:

| Dataset            | N_total  | N_train | N_validate | N_test | x_num | x_cat | x_encoded | x_total |
|--------------------|----------|---------|------------|--------|-------|-------|-----------|---------|
| Adult              | 32,561   | 22,792  | 3,256      | 6,513  | 6     | 8     | 99        | 105     |
| Electricity        | 45,312   | 31,717  | 4,532      | 9,063  | 7     | 1     | 7         | 14      |
| COMPAS             | 16,644   | 11,650  | 1,665      | 3,329  | 8     | 8     | 50        | 58      |
| Higgs              | 1,000,000| 700,000 | 100,000    | 200,000| 28    | 0     | 0         | 28      |
| house_16H          | 22,784   | 15,948  | 2,279      | 4,557  | 16    | 0     | 0         | 16      |
| jm1                | 10,885   | 7,619   | 1,089      | 2,177  | 21    | 0     | 0         | 21      |
| BreastCancer       | 569      | 398     | 57         | 114    | 30    | 0     | 0         | 30      |
| WineQuality-White  | 4,898    | 3,428   | 490        | 980    | 11    | 0     | 0         | 11      |
| WineQuality-Red    | 1,599    | 1,119   | 160        | 320    | 11    | 0     | 0         | 11      |
| phoneme            | 5,404    | 3,782   | 541        | 1,081  | 5     | 0     | 0         | 5       |
| MiniBooNE          | 130,064  | 91,044  | 13,007     | 26,013 | 50    | 0     | 0         | 50      |

## Model Information

The benchmark evaluates adversarial attacks across four types of predictive models:

- **Logistic Regression**: A linear model commonly used for binary and multiclass classification tasks. It serves as a strong baseline for tabular data.
- **MLP (Multilayer Perceptron)**: A feedforward neural network with one or more hidden layers, capable of modeling complex nonlinear relationships in tabular data.
- **TabTransformer**: A transformer-based architecture designed specifically for tabular data, leveraging attention mechanisms to capture feature interactions, especially for categorical features. [Reference](https://github.com/lucidrains/tab-transformer-pytorch)
- **FTTransformer**: An efficient transformer variant for tabular data, focusing on fast training and inference while maintaining high accuracy. [Reference](https://github.com/lucidrains/tab-transformer-pytorch)

## Attack Methods

### Bounded Attacks
- **FGSM (Fast Gradient Sign Method):** One-step, uses the sign of the gradient.
- **BIM (Basic Iterative Method):** Multi-step, iterative FGSM.
- **PGD (Projected Gradient Descent):** Multi-step, generalizes BIM.

All bounded attacks constrain the perturbation $\delta$ by a budget $\epsilon$:

$$ x^{adv} = x + \delta, \quad \|\delta\|_p \leq \epsilon $$

### Unbounded Attacks
- **C&W (Carlini & Wagner):** Minimizes $L_2$ perturbation plus a confidence term.
- **DeepFool:** Finds the closest decision boundary under a linear approximation.

Unbounded attacks minimize the distance between $x$ and $x^{adv}$, subject to misclassification:

$$ \min_{\delta} \|\delta\|_p, \quad \text{subject to } f(x + \delta) \neq f(x) $$

### Random Noise Baselines
- **Linf Gaussian Noise:** Gaussian noise clipped by $\epsilon$.

## Results

Results are available in the [result page](./results.html).

### How to Analyze the Results
- Compare different attack methods on a single predictive model (with varying perturbation budgets).
- Compare different predictive models for a single attack method.

## Acknowledgements

We would like to thank these public repositories from which we have borrowed code and ideas:

- **Predictive Models**
    - Implementation of TabTransformer and FTtransformer: https://github.com/lucidrains/tab-transformer-pytorch
- **Attack Methods**
    - Foolbox: https://github.com/bethgelab/foolbox