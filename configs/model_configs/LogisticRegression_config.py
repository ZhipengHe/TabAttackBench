import numpy as np


config_random = {
    "model_name": {
        "value": "LogisticRegression"
    },
    # parameters space used for random search
    # following WandB sweeps syntax
    "model_type": {
        "value": "sklearn"
    },
    "model__penalty": { 
        "value": [None, 'l1', 'l2', 'elasticnet']
    },
    "model__C": { 
        "values": [1e-3, 1e-2, 1e-1, 1, 10, 100]
    },
    "model__solver": { 
        "value": 'saga',
    },
    "use_gpu": {
        "value": False
    },
    # add default parameters specific to classification situations
    # ...
}

config_default = {
    "model_name": {
        "value": "LogisticRegression"
    },
    # Default parameter values
    "model_type": {
        "value": "sklearn" 
    },
    "use_gpu": {
        "value": False
    },
    "one_hot_encoder": {
        "value": True
    }
}
