import wandb
import torch
from torch import nn, optim
from tqdm import tqdm
import os



def train(model, train_data, val_data, criterion, optimizer, config, wandb_run=None):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!

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
        if wandb_run:
            wandb_run.log(metrics)


def train_batch(batch_X, batch_y, model, optimizer, criterion):    
    # Forward pass ➡
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)

    ## Used for BCEWithLogitsLoss
    # # Calculate the number of correct predictions
    # predicted = (outputs >= 0.5).float()
    # correct = (predicted == batch_y).sum().item()
    
    # Used for CrossEntropyLoss
    # Get the class with the highest probability
    _, predicted_classes = outputs.max(dim=1)  
    correct = (predicted_classes == batch_y).sum().item()

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

    ## Used for BCEWithLogitsLoss
    # # Calculate the number of correct predictions
    # predicted = (outputs >= 0.5).float()
    # correct = (predicted == batch_y).sum().item()

    # Used for CrossEntropyLoss
    # Get the class with the highest probability
    _, predicted_classes = outputs.max(dim=1)  
    correct = (predicted_classes == batch_y).sum().item()

    return loss, correct


# Evaluate the model on the testing data
def test(model, data, config, stage="test", wandb_run=None):
    model_name = config["model"]
    data_name = config["dataset"]
    X_test_tensor, y_test_tensor = data
    run_name = wandb.run.name if wandb_run else "test"
    sweep_id = wandb.run.sweep_id if wandb_run and stage == "sweep" else stage

    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test_tensor)
        # predicted = (test_outputs >= 0.5).float()
        _, predicted = test_outputs.max(dim=1)
        accuracy = (predicted == y_test_tensor).float().mean()
        print(f"Accuracy: {accuracy.item() * 100:.2f}%")
        
        if wandb_run:
            wandb_run.log({"test_accuracy": accuracy})
        else:
            print("Test results logged to WandB not available. Consider passing a WandB run object for logging.")

    if not os.path.exists(f"models/{stage}/{model_name}/{data_name}/"):
    # If it doesn't exist, create it
        os.makedirs(f"models/{stage}/{model_name}/{data_name}/")

    # Save the model in the exchangeable ONNX format
    onnx_file_name = f"models/{stage}/{model_name}/{data_name}/" + f"{sweep_id}_run-{run_name}.onnx"
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