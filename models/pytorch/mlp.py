"""MLP model architecture for tabular data."""

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
        """
        super().__init__()

        self.num_categorical_feature = num_categorical_feature  # Added
        # print(f'{self.num_categorical_feature=}')

        if categories is not None:
            input_dim += len(categories) * embedding_dim - len(categories)
            # print(f'{input_dim=}')
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(
                sum(categories), embedding_dim)
            nn.init.kaiming_uniform_(
                self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

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

        if not self.num_categorical_feature is None:
            x_cat = x[:, :self.num_categorical_feature].long()
            x_num = x[:, self.num_categorical_feature:]
        else:
            x_cat = None
            x_num = x
            
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x_cat_emb = self.category_embeddings(x_cat + self.category_offsets[None]).view(x_cat.size(0), -1)
            x.append(x_cat_emb)
        x = torch.cat(x, dim=-1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


def train_log(loss, epoch, batch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss, "batch": batch})


def train(model, data, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0

    X_train_tensor, y_train_tensor = data
    for epoch in tqdm(range(config["epochs"])):
        total_correct = 0  # Reset total_correct for each epoch
        for i in range(0, X_train_tensor.size(0), config["batch_size"]):

            batch_X = X_train_tensor[i:i+config["batch_size"]]
            batch_y = y_train_tensor[i:i+config["batch_size"]]

            loss, correct = train_batch(batch_X, batch_y, model, optimizer, criterion)
            total_correct += correct  # Accumulate correct predictions
            example_ct +=  batch_X.size(0)
            batch_ct += 1

            metrics = {"epoch": epoch, "loss": loss, "batch": batch_ct}

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                wandb.log(metrics)
        epoch_accuracy = total_correct / len(X_train_tensor)
        wandb.log({**metrics, "accuracy": epoch_accuracy})


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


    # Evaluate the model on the testing data


def test(model, data):
    X_test_tensor, y_test_tensor = data

    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test_tensor)
        predicted = (test_outputs >= 0).float()
        accuracy = (predicted == y_test_tensor).float().mean()
        print(f"Accuracy: {accuracy.item() * 100:.2f}%")
        
        wandb.log({"test_accuracy": accuracy})

    # Save the model in the exchangeable ONNX format
    # torch.onnx.export(model, X_test_tensor, "model.onnx")
    # wandb.save("model.onnx")

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

    # Create a list of boolean values indicating whether each feature is categorical
    categorical_indicator = [col in categorical_cols for col in X.columns]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardize numerical features (optional but recommended)
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    sweep_config = {
        'method': 'grid'
        }

    parameters_dict = {
        'epochs': {
            'values': [20]
            },
        'optimizer': {
            'values': ['adam']
            },
        'dropout': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            },
        'embedding_dim': {
            'values': [2, 4, 8]
            },
        'learning_rate': {
            'values': [0.001, 0.01, 0.1]
            },
        'batch_size': {
            'values': [32, 64, 128, 256]
            },
        }

    # Create an instance of the MLP model with appropriate hyperparameters
    input_dim = X_train.shape[1]
    output_dim = 1  # Binary classification
    hidden_layer_dims = [64,32,16,8]
    # dropout = 0.2
    # embedding_dim = 4

    config = dict(
        # epochs=10,
        classes=2,
        kernels=hidden_layer_dims,
        # batch_size=128,
        # learning_rate=0.001,
        dataset="Adult",
        architecture="MLP")

    sweep_config['parameters'] = parameters_dict

    import pprint

    pprint.pprint(sweep_config)



    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")


    
    print(f'{config=}')

    def main():
        with wandb.init(project="pytorch-mlp-demo"):
            # access all HPs through wandb.config, so logging matches execution!

            config = wandb.config


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
            train(model, (X_train_tensor, y_train_tensor), criterion, optimizer, config)


            # and test its final performance
            test(model, (X_test_tensor, y_test_tensor))
    
    wandb.agent(sweep_id, function=main)


    # # Evaluate the model on the testing data
    # with torch.no_grad():
    #     model.eval()
    #     test_outputs = model(X_test_tensor)
    #     predicted = (test_outputs >= 0).float()
    #     accuracy = (predicted == y_test_tensor).float().mean()
    #     print(f"Accuracy: {accuracy.item() * 100:.2f}%")
