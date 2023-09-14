# %% [markdown]
# ## Task 3: Model Training

# %%
import wandb
import os
os.environ["WANDB_NOTEBOOK_NAME"] = "model_training.ipynb"
wandb.login()

# %%
import torch
from torch import optim, nn
from tqdm import tqdm

from models import run_pytorch
from data import get_datasets

from models.pytorch.mlp import MLP
from models.pytorch.tab_transformer import TabTransformer
from models.pytorch.ft_transformer import FTTransformer
from models.jax.logistic_regression import LogisticRegression


# %%
def model_config(model, input_dim, output_dim, categories_list, numerical_cols, device):
    if model == "MLP":
        model_config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "num_hidden_layers": 2,
            "hidden_layer_dims": [64, 32],
            "dropout": 0.2,
            "categories": categories_list,
            "embedding_dim": 8,
            "num_categorical_feature": len(categories_list),
        }
        train_config = {
            "epochs": 30,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "model": "MLP",
            "dropout": 0.2,
        }
        return MLP(**model_config).to(device), train_config
    
    elif model == "TabTransformer":
        model_config = {
            "categories": categories_list,
            "num_continuous": len(numerical_cols),
            "dim": 8, # can sweep
            "dim_out": output_dim,
            "depth": 6,
            "heads": 8,
            "attn_dropout": 0.2,
            "ff_dropout": 0.2,
            "mlp_hidden_mults": (4, 2), 
            "mlp_act": nn.ReLU(),
            "continuous_mean_std": None,
        }

        train_config = {
            "epochs": 30,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "model": "TabTransformer",
            "dropout": 0.2,
        }
        return TabTransformer(**model_config).to(device), train_config

    elif model == "FTTransformer":
        model_config = {
            "categories": categories_list,
            "num_continuous": len(numerical_cols),
            "dim": 8, 
            "dim_out": output_dim,
            "depth": 6,
            "heads": 8, 
            "attn_dropout": 0.2, 
            "ff_dropout": 0.2, 
        }

        train_config = {
            "epochs": 30,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "model": "FTTransformer",
            "dropout": 0.2,
        }
        return FTTransformer(**model_config).to(device), train_config


    # elif model_name == "TabNet":
    #     model_config = {}

    # return model

if __name__ == "__main__":
    wandb_run = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in ["Adult", "Electricity", "Mushroom"]: # "Adult", "Electricity", "Higgs", "KDDCup09_appetency", "Mushroom"
        X_train, y_train, X_val, y_val, X_test, y_test, \
            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_val_tensor, y_val_tensor, \
            info = get_datasets.get_dataset(dataset_name, device)
        for model_name in ["MLP", "TabTransformer", "FTTransformer"]: # "MLP", "TabTransformer", "FTTransformer"
            model, train_config = model_config(model_name, X_train.shape[1], 2, info["categories_list"], info["numerical_cols"], device)
            train_config["dataset"] = dataset_name
            criterion = nn.CrossEntropyLoss()
            optimizer = run_pytorch.build_optimizer(model, "adam", train_config["learning_rate"])

            config = {**train_config}
            print(f"Training model: {model_name} on dataset: {dataset_name}")
            if wandb_run:
                with wandb.init(project="TabAttackBench-ModelTraining", config=config):
                    run_pytorch.train(model, (X_train_tensor, y_train_tensor), (X_val_tensor, y_val_tensor), criterion, optimizer, train_config, wandb_run=wandb.run)
                    # and test its final performance
                    run_pytorch.test(model, (X_test_tensor, y_test_tensor), train_config, stage="train", wandb_run=wandb.run)
            else:
                run_pytorch.train(model, (X_train_tensor, y_train_tensor), (X_val_tensor, y_val_tensor), criterion, optimizer, train_config)
                run_pytorch.test(model, (X_test_tensor, y_test_tensor), train_config, stage="train")
