"""
Configuration file for all models used in the project. This file is inspired by the configuration file in the following repository:
https://github.com/LeoGrin/tabular-benchmark/blob/main/src/configs/all_model_configs.py
"""

import sys
# sys.path.append("..") # Adds higher directory to python modules path.

# Importing all models used in the project
from xgboost import XGBClassifier # XGBoost


total_config = {}
model_keyword_dic = {}

# # Template for adding a model
# # ADD YOU MODEL HERE ##
# from configs.model_configs.your_file import config_default, config_random #replace template.py by your parameters
# keyword = "your_model"
# total_config[keyword] = {
#     "random": config_random,
#     "default": config_default
#     }
# #these constructor should create an object
# # with fit and predict methods
# model_keyword_dic[config_classif["model_name"]] = YourModelClassClassifier
# ############################


# XGBoost Classifier

from configs.model_configs.XGBoost_config import config_default, config_random
keyword = "XGBoost"
total_config[keyword] = {
    "random": config_random,
    "default": config_default
    }

model_keyword_dic[config_random["model_name"]["value"]] = XGBClassifier