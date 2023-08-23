## Configurations

This directory contains the configuration files for the different models, datasets, attacks, etc.

### Models

Config parameters for the models are stored in the `model_configs`. Each model has its own config file. The config files are named after the model they configure. For example, the config file for the `XGBoost` model is `xgb_config.py`.

All available parameters are listed in the below. Each model has both default parameters and model-specific parameters. The default parameters are shared by all models:

- `model_name`: Name of the model, such as `XGBoost`, `LogisticRegression`, `MLP`, etc.
- `model_type`: Type of the model, such as `sklearn`, `keras`, `pytorch`, etc.
- `use_gpu`: Whether to use GPU for training, such as `True` or `False`.
- `one_hot_encoder`: Whether to use one-hot encoder for categorical features, such as `True` or `False`.
- ...

The model-specific parameters are specific to the model, which start with `model__`. Here provide a brief description of model-specific parameters for the models. We use WandB sweep config to define the search space for the hyperparameters. For more information about WandB sweep config, please refer to [WandB sweep config](https://docs.wandb.ai/guides/sweeps/configuration).

#### Logistic Regression (sklearn)

Reference: [Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

- `model__penalty`: Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. (default: 'l2')
    - Sweep config:
        ```python
        "model__penalty": { 
            "value": [None, 'l1', 'l2', 'elasticnet']
        },
        ```

- `model__C`: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization. (default: 1.0)
    - Sweep config:
        ```python
        "model__C": { 
            "values": [1e-3, 1e-2, 1e-1, 1, 10, 100]
        },
        ```

- `model__solver`: Algorithm to use in the optimization problem. (default: 'lbfgs')
    - Sweep config:
        ```python
        "model__solver": { 
            "value": 'saga',
        },
        ```

#### XGBoost (sklearn)

Reference: [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster)

- `model__n_estimators`: Number of gradient boosted trees. Equivalent to number of boosting rounds. (default: 100)
    - Sweep config:
        ```python
        "model__n_estimators": {
            "values": 1000
        }
        ```
- `model__learning_rate`: Boosting learning rate (xgb’s “eta”) (default: 0.3)
    - Sweep config: 
        ```python
        "model__learning_rate": {
            'distribution': "log_uniform_values",
            'min': 1E-5, 
            'max': 0.7,
        }
        ```
- `model__gamma`: Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. (default: 0)
    - Sweep config: 
        ```python
        "model__gamma": {
            "distribution": "log_uniform_values",
            'min': 1E-8, 
            'max': 7,
        }
        ```
- `model__max_depth`: Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit, limit is required for depth-wise grow policy. (default: 6)
    - Sweep config: 
        ```python
        "model__max_depth": {
            'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        }
        ```
- `model__min_child_weight`: Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be. (default: 1)
    - Sweep config: 
        ```python
        "model__min_child_weight": {
            "distribution": "q_log_uniform_values",
            'min': 1,
            'max': 100,
            'q': 1,
        }
        ```
- `model__subsample`: Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting. (default: 1)
    - Sweep config: 
        ```python
        "model__subsample": {
            "distribution": "uniform",
            'min': 0.5,
            'max': 1.0,
        }
        ```
- `model__colsample_bytree`: Subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed. (default: 1)
    - Sweep config: 
        ```python
        "model__colsample_bytree": {
            "distribution": "uniform",
            'min': 0.5,
            'max': 1.0,
        }
        ```
- `model__colsample_bylevel`: Subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree. (default: 1)
    - Sweep config: 
        ```python
        "model__colsample_bylevel": {
            "distribution": "uniform",
            'min': 0.5,
            'max': 1.0,
        }
        ```
- `model__use_label_encoder`: perform label encoding on categorical variables into integers (default: False)
    - Sweep config: 
        ```python
        "model__use_label_encoder": {
            "value": False
        },
        ```
- `model__reg_alpha`: L1 regularization term on weights. Increasing this value will make model more conservative. (default: 0)
    - Sweep config: 
        ```python
        "model__reg_alpha": {
            "distribution": "log_uniform_values",
            'min': 1E-8, 
            'max': 1E2,
        }
        ```
- `model__reg_lambda`: L2 regularization term on weights. Increasing this value will make model more conservative. (default: 1)
    - Sweep config: 
        ```python
        "model__reg_lambda": {
            "distribution": "log_uniform_values",
            'min': 1,
            'max': 4,
        }
        ```
- `model__early_stopping_rounds`: Activates early stopping. Validation metric needs to improve at least once in every `early_stopping_rounds` round(s) to continue training. (default: None)
    - Sweep config: 
        ```python
        "model__early_stopping_rounds": {
            "values": 20
        }
        ```

