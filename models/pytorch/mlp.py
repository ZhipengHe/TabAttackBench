"""MLP model architecture for tabular data.
TODO: 
1. move train and test to a separate file
2. move data loading to a separate file
"""

import os
import random
import math
from typing import Optional, List

import numpy as np
# import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import requests
from io import StringIO

import itertools

from tqdm import tqdm
import wandb
wandb.login()


class MLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_hidden_layers: int,
        hidden_layer_dims: List[int],
        dropout: float,
        output_dim: int,
        categories: Optional[List[int]],
        embedding_dim: int,
        num_categorical_feature: int,
        num_numerical_feature: int,
    ) -> None:
        """MLP model architecture for tabular data.
        
        Args:
            input_dim (int): Number of input features.
            num_hidden_layers (int): Number of hidden layers. #TODO: Remove this
            hidden_layer_dims (List[int]): List of hidden layer dimensions.
            dropout (float): Dropout rate.
            output_dim (int): Number of output classes.
            categories (Optional[List[int]]): List of number of unique categories for each categorical feature.
            embedding_dim (int): Embedding dimension for categorical features.
            num_categorical_feature (int): Number of categorical features.
            num_numerical_feature (int): Number of numerical features.
        """
        super().__init__()

        self.num_numerical_feature = num_numerical_feature
        self.num_categorical_feature = num_categorical_feature  # Added
        self.categories = categories

        if categories is not None:
            # input_dim += len(categories) * embedding_dim - len(categories)
            # # print(f'{input_dim=}')
            # category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            # self.register_buffer('category_offsets', category_offsets)
            # self.category_embeddings = nn.Embedding(
            #     sum(categories), embedding_dim)
            # print(f'{self.category_embeddings.weight.shape=}')

            ## One embedding for each categorical feature
            self.embedding = nn.ModuleList([nn.Embedding(cat_dim, embedding_dim) for cat_dim in categories])
            # input_dim = (embedding_dim * num_categorical_feature) + num_numerical_feature
            input_dim = (embedding_dim * sum(categories)) + num_numerical_feature

        # Create a list for all layers (input + hidden + output)
        self.layers = nn.ModuleList()
        
        # Add the input layer
        self.layers.append(nn.Linear(input_dim, hidden_layer_dims[0]))
        
        # Add hidden layers
        for i in range(1, len(hidden_layer_dims)):
            self.layers.append(nn.Linear(hidden_layer_dims[i-1], hidden_layer_dims[i]))
        
        self.dropout = dropout
        self.head = nn.Linear(
            hidden_layer_dims[-1] if hidden_layer_dims else input_dim, output_dim)

    def forward(self, x):
        # from data preprocessing, numerical features are first and categorical features are last
        if not self.num_categorical_feature is None:
            x_num = x[:, :self.num_numerical_feature]
            x_cat = x[:, self.num_numerical_feature:].long()
            # print(f"{x_num.shape=}")
            # print(f"{x_cat.shape=}")

            ## One embedding for each categorical feature
            x_cat_emb = []
            start = 0
            for i, item in enumerate(self.categories):
                embedding = self.embedding[i](x_cat[:, start:start+item]).view(x_cat[:, start:start+item].size(0), -1)
                x_cat_emb.append(embedding)
                # print(f"{x_cat_emb[-1].shape=}")
                start += item
            
            x_cat_emb_flat = torch.cat(x_cat_emb, dim=-1)
            # print(f"{x_cat_emb_flat.shape=}")
            x = torch.cat([x_num, x_cat_emb_flat], dim=-1)
            
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        return x


def train_log(loss, epoch, batch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss, "batch": batch})


def train(model, train_data, val_data, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    # wandb.watch(model, criterion, log="all", log_freq=10)

    # Separate train and validation data
    X_train_tensor, y_train_tensor = train_data
    X_val_tensor, y_val_tensor = val_data

    for epoch in tqdm(range(config["epochs"])):
        # Training phase
        total_correct_train = 0
        total_loss_train = 0.0
        model.train()  # Set model to training mode

        for i in range(0, X_train_tensor.size(0), config["batch_size"]):
            batch_X = X_train_tensor[i:i+config["batch_size"]]
            batch_y = y_train_tensor[i:i+config["batch_size"]]

            loss, correct = train_batch(batch_X, batch_y, model, optimizer, criterion)
            total_loss_train += loss.item()
            total_correct_train += correct

        # Validation phase
        total_correct_val = 0
        total_loss_val = 0.0
        model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for i in range(0, X_val_tensor.size(0), config["batch_size"]):
                batch_X_val = X_val_tensor[i:i+config["batch_size"]]
                batch_y_val = y_val_tensor[i:i+config["batch_size"]]

                val_loss, val_correct = evaluate_batch(batch_X_val, batch_y_val, model, criterion)
                total_loss_val += val_loss.item()
                total_correct_val += val_correct

        # Calculate and log metrics for both training and validation
        train_accuracy = total_correct_train / len(X_train_tensor)
        val_accuracy = total_correct_val / len(X_val_tensor)

        train_loss_avg = total_loss_train / (len(X_train_tensor) / config["batch_size"])
        val_loss_avg = total_loss_val / (len(X_val_tensor) / config["batch_size"])

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss_avg,
            "val_accuracy": val_accuracy
        }

        wandb.log(metrics)


def train_batch(batch_X, batch_y, model, optimizer, criterion):    
    # Forward pass ➡
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)

    # Calculate the number of correct predictions
    predicted = (outputs >= 0.5).float()
    correct = (predicted == batch_y).sum().item()
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss, correct

def evaluate_batch(batch_X, batch_y, model, criterion):
    # Forward pass ➡
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)

    # Calculate the number of correct predictions
    predicted = (outputs >= 0.5).float()
    correct = (predicted == batch_y).sum().item()

    return loss, correct

    # Evaluate the model on the testing data


def test(model, data, sweep_id, wandb_run=None):
    X_test_tensor, y_test_tensor = data
    run_name = wandb.run.name if wandb_run else "test"

    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test_tensor)
        predicted = (test_outputs >= 0.5).float()
        accuracy = (predicted == y_test_tensor).float().mean()
        print(f"Accuracy: {accuracy.item() * 100:.2f}%")
        
        if wandb_run:
            wandb_run.log({"test_accuracy": accuracy})
        else:
            print("Test results logged to WandB not available. Consider passing a WandB run object for logging.")

    # Save the model in the exchangeable ONNX format
    onnx_file_name = f"models/sweeps/" + f"sweep-run-{run_name}.onnx"
    torch.onnx.export(model, X_test_tensor, onnx_file_name)
    
    if wandb_run:
        wandb_run.save(onnx_file_name)
    else:
        print(f"ONNX model saved as {onnx_file_name}. Consider passing a WandB run object for saving.")


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


if __name__ == '__main__':

    # Import your MLP class definition here

    # Define the URL for the Adult dataset on UCI Machine Learning Repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    # Define column names for the dataset
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"
    ]

    # Fetch the dataset from the URL
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text), header=None,
                       names=column_names, sep=',\s*', engine='python')

    # Define categorical and numerical columns
    categorical_cols = ["workclass", "education", "marital_status",
                        "occupation", "relationship", "race", "sex", "native_country"]
    numerical_cols = ["age", "fnlwgt", "education_num",
                      "capital_gain", "capital_loss", "hours_per_week"]

    # Create a dictionary to store the number of unique categories for each categorical column
    categories = {}

    # Use LabelEncoder to encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Encode the target variable 'income' as 1 and 0
    data['income'] = data['income'].map({'>50K': 1, '<=50K': 0})

    # Calculate the number of unique categories for each categorical column
    for col in categorical_cols:
        unique_values = data[col].nunique()
        categories[col] = unique_values

    # Convert the dictionary to a list of values
    categories_list = list(categories.values())

    # Separate features (X) and target (y)
    X = data[categorical_cols + numerical_cols]
    y = data["income"]

    # Split the dataset into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

    # Standardize numerical features (optional but recommended)
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Convert the data to PyTorch tensors
    X_train = X_train.values
    y_train = y_train.values
    X_val = X_val.values
    y_val = y_val.values
    X_test = X_test.values
    y_test = y_test.values

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    sweep_config = {
        'method': 'grid'
        }

    parameters_dict = {
        'architecture': {
            'values': ['MLP']
        },
        'epochs': {
            'values': [20]
            },
        'optimizer': {
            'values': ['adam']
            },
        'dropout': {
            'values': [0.0]
            },
        'embedding_dim': {
            'values': [4]
            },
        'learning_rate': {
            'values': [0.001]
            },
        'batch_size': {
            'values': [256]
            },
        }

    # Create an instance of the MLP model with appropriate hyperparameters
    input_dim = X_train.shape[1]
    output_dim = 1  # Binary classification
    hidden_layer_dims = [64,32]

    sweep_config['parameters'] = parameters_dict

    import pprint

    pprint.pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    # print(f'{config=}')

    def main():
        with wandb.init(project="pytorch-mlp-demo"):
            # access all HPs through wandb.config, so logging matches execution!
            config = wandb.config

            print(wandb.run.sweep_id)


            model = MLP(
                input_dim=input_dim,
                num_hidden_layers=len(hidden_layer_dims),
                hidden_layer_dims=hidden_layer_dims,
                dropout=config.dropout,
                output_dim=output_dim,
                categories=categories_list,
                embedding_dim=config.embedding_dim,
                # Provide the list of categorical columns
                num_categorical_feature=len(categorical_cols),
            )

            # Define a loss function and an optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = build_optimizer(model, config.optimizer, config.learning_rate)

            # and use them to train the model
            train(model, (X_train_tensor, y_train_tensor), (X_val_tensor, y_val_tensor), criterion, optimizer, config)

            # and test its final performance
            test(model, (X_test_tensor, y_test_tensor), config, wandb_run=wandb.run)
    
    wandb.agent(sweep_id, function=main)

