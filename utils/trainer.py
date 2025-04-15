import collections
import datetime
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Run one epoch of training.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        device: Device to run on

    Returns:
        dict: Dictionary containing training metrics
    """
    model.train()
    loss_train = 0.0
    correct_train = 0
    total_train = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)

        outputs = model(imgs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        total_train += labels.shape[0]
        correct_train += int((predicted == labels).sum())

    # Calculate training metrics
    train_loss = loss_train / len(train_loader)
    train_accuracy = correct_train / total_train

    return {"loss": train_loss, "accuracy": train_accuracy}


@contextmanager
def create_summary_writer(model_name, hyperparams=None):
    """Context manager for creating and managing a TensorBoard SummaryWriter.

    Args:
        model_name (str): Name of the model for the log directory
        hyperparams (dict, optional): Hyperparameters to log

    Yields:
        SummaryWriter: TensorBoard writer
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_architecture = (
        hyperparams.get("model_architecture", "") if hyperparams else ""
    )
    log_dir = f"runs/{model_name}_{model_architecture}_{timestamp}"
    writer = SummaryWriter(log_dir)

    print(f"TensorBoard logs will be saved to {log_dir}")

    if hyperparams:
        # Log hyperparameters as text
        param_str = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        writer.add_text("Hyperparameters", param_str)

    try:
        yield writer
    finally:
        writer.close()
        print(f"TensorBoard writer closed for {log_dir}")


def evaluate_model(model, val_loader, loss_fn, device, class_names=None):
    """Evaluate the model on validation data.

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run on
        class_names (list, optional): List of class names

    Returns:
        dict: Dictionary containing validation metrics and predictions
    """
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total_val += labels.shape[0]
            correct_val += int((predicted == labels).sum())

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate validation metrics
    val_loss = val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_f1 = f1_score(all_labels, all_preds, average="weighted")

    return {
        "loss": val_loss,
        "accuracy": val_accuracy,
        "f1": val_f1,
        "predictions": all_preds,
        "true_labels": all_labels,
    }


def log_metrics(writer, train_metrics, val_metrics, optimizer, epoch):
    """Log metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        train_metrics (dict): Training metrics
        val_metrics (dict): Validation metrics
        optimizer: PyTorch optimizer
        epoch (int): Current epoch
    """
    # Log scalar metrics
    writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
    writer.add_scalar("Loss/validation", val_metrics["loss"], epoch)
    writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
    writer.add_scalar("Accuracy/validation", val_metrics["accuracy"], epoch)
    writer.add_scalar("F1/validation", val_metrics["f1"], epoch)

    # Log learning rate
    writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)


def log_model_info(writer, model, epoch, train_loader, device):
    """Log model parameters and gradients to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch model
        epoch (int): Current epoch
        train_loader: Training data loader
        device: Device to run on
    """
    # Log model graph (only once)
    if epoch == 1:
        example_images, _ = next(iter(train_loader))
        try:
            writer.add_graph(model, example_images.to(device))
        except Exception as e:
            print(f"Failed to add model graph to TensorBoard: {e}")

    # Log histograms of model parameters
    for name, param in model.named_parameters():
        writer.add_histogram(f"Parameters/{name}", param, epoch)
        if param.grad is not None:
            writer.add_histogram(f"Gradients/{name}", param.grad, epoch)


def log_predictions(
    writer, model, val_loader, device, class_names, epoch, num_images=10
):
    """Log prediction visualizations to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to run on
        class_names (list): List of class names
        epoch (int): Current epoch
        num_images (int): Number of images to visualize
    """
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            for j in range(imgs.size()[0]):
                if images_so_far >= num_images:
                    break

                ax = plt.subplot(2, num_images // 2, images_so_far + 1)
                ax.axis("off")
                ax.set_title(
                    f"pred: {class_names[preds[j]]}\ntrue: {class_names[labels[j]]}",
                    color=("green" if preds[j] == labels[j] else "red"),
                )

                # Denormalize and convert to numpy for matplotlib
                img = imgs[j].cpu().numpy().transpose((1, 2, 0))
                mean = np.array([0.4915, 0.4823, 0.4468])
                std = np.array([0.2470, 0.2435, 0.2616])
                img = std * img + mean
                img = np.clip(img, 0, 1)

                plt.imshow(img)
                images_so_far += 1
                if images_so_far >= num_images:
                    break

    writer.add_figure(f"Predictions/Epoch_{epoch}", fig, epoch)
    plt.close(fig)


def log_embeddings(writer, model, val_loader, device, class_names, n_epochs):
    """Log embeddings to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to run on
        class_names (list): List of class names
        n_epochs (int): Total number of epochs
    """
    features = []
    labels_list = []

    # Get features from the last layer before classification
    def hook_fn(module, input, output):
        features.append(input[0].cpu().numpy())

    # Register hook to the second-to-last layer
    try:
        handle = model.fc1.register_forward_hook(hook_fn)

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                model(imgs)
                labels_list.extend(labels.numpy())

        handle.remove()

        # Concatenate all features
        features = np.concatenate(features)

        # Select a subset of data for visualization (max 10000 points)
        max_samples = min(10000, len(features))
        indices = np.random.choice(len(features), max_samples, replace=False)

        # Log embeddings
        writer.add_embedding(
            features[indices],
            metadata=[class_names[l] for l in np.array(labels_list)[indices]],
            label_img=None,
            global_step=n_epochs,
        )
    except Exception as e:
        print(f"Failed to log embeddings to TensorBoard: {e}")


class EarlyStopping:
    """Early stopping to terminate training when validation loss doesn't improve.

    Args:
        patience (int): How many epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as an improvement.
        mode (str): 'min' for monitoring metrics that decrease (like loss),
                    'max' for metrics that increase (like accuracy).
        verbose (bool): If True, prints a message for each improvement.
    """

    def __init__(self, patience=10, min_delta=0.0, mode="min", verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        # Set the direction based on mode
        self.monitor_op = np.less if mode == "min" else np.greater
        self.min_delta = min_delta if mode == "min" else -min_delta

    def __call__(self, epoch, current_score, model=None, path=None):
        """Check if training should be stopped.

        Args:
            epoch (int): Current epoch number
            current_score (float): Current validation metric to monitor
            model (torch.nn.Module, optional): Model to save if score improves
            path (str, optional): Path to save the model

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            # First epoch
            self.best_score = current_score
            self.best_epoch = epoch
            self.save_checkpoint(current_score, model, path)
        elif self.monitor_op(current_score - self.min_delta, self.best_score):
            # Score improved
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(current_score, model, path)
        else:
            # Score did not improve
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, score, model, path):
        """Save model when validation score improves."""
        if self.verbose:
            improved = "improved" if self.best_score == score else "did not improve"
            metric_name = "loss" if self.mode == "min" else "score"
            print(
                f"Validation {metric_name} {improved} ({self.best_score:.6f} --> {score:.6f})"
            )

        if model is not None and path is not None:
            torch.save(model.state_dict(), path)
            if self.verbose:
                print(f"Model saved to {path}")


def training_loop(
    n_epochs,
    optimizer,
    model,
    loss_fn,
    train_loader,
    val_loader,
    epoch_trainer,
    device,
    class_names,
    model_name="model",
    early_stopping_params=None,
):
    """Main training loop with TensorBoard logging and early stopping.

    Args:
        n_epochs (int): Maximum number of epochs
        optimizer: PyTorch optimizer
        model: PyTorch model
        loss_fn: Loss function
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on
        class_names (list): List of class names
        model_name (str): Name for the model in logs
        early_stopping_params (dict, optional): Parameters for early stopping
            {
                'patience': int,
                'min_delta': float,
                'metric': str ('loss', 'accuracy', or 'f1'),
                'mode': str ('min' or 'max')
            }

    Returns:
        dict: Dictionary containing best metrics
    """
    print(f"Training on device {device}")

    # Move model to device
    model.to(device=device)

    # Setup early stopping if parameters are provided
    early_stopping = None
    if early_stopping_params:
        metric = early_stopping_params.get("metric", "f1")
        mode = early_stopping_params.get("mode", "min" if metric == "loss" else "max")
        patience = early_stopping_params.get("patience", 10)
        min_delta = early_stopping_params.get("min_delta", 0.0)
        verbose = early_stopping_params.get("verbose", False)

        early_stopping = EarlyStopping(
            patience=patience, min_delta=min_delta, mode=mode, verbose=verbose
        )

        print(
            f"Early stopping enabled: monitoring {metric}, mode={mode}, patience={patience}"
        )

    # Create hyperparameters dict for logging
    hyperparams = {
        "batch_size": train_loader.batch_size,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
        "epochs": n_epochs,
        "optimizer": optimizer.__class__.__name__,
        "model_architecture": model.__class__.__name__,
    }

    if early_stopping_params:
        hyperparams.update(
            {
                "early_stopping_metric": early_stopping_params.get("metric", "loss"),
                "early_stopping_patience": early_stopping_params.get("patience", 10),
                "early_stopping_min_delta": early_stopping_params.get("min_delta", 0.0),
            }
        )

    # Create TensorBoard writer using context manager
    with create_summary_writer(model_name, hyperparams) as writer:
        best_val_f1 = 0.0
        best_metrics = {}

        for epoch in range(1, n_epochs + 1):
            # Train for one epoch
            train_metrics = epoch_trainer(
                model, train_loader, optimizer, loss_fn, device
            )

            # Evaluate the model
            val_metrics = evaluate_model(
                model, val_loader, loss_fn, device, class_names
            )

            # Log metrics to TensorBoard
            log_metrics(writer, train_metrics, val_metrics, optimizer, epoch)

            # Log model information periodically
            if epoch % 10 == 0 or epoch == 1:
                log_model_info(writer, model, epoch, train_loader, device)
                log_predictions(writer, model, val_loader, device, class_names, epoch)

            # Print metrics periodically
            if epoch == 1 or epoch % 5 == 0:
                print(f"{datetime.datetime.now()} Epoch {epoch}")
                print(
                    f"  Training:   Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}"
                )
                print(
                    f"  Validation: Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}"
                )

            # Save best model based on F1 score
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_metrics = val_metrics.copy()
                torch.save(model.state_dict(), f"runs/{model_name}_best_model.pth")

            # Check early stopping condition
            if early_stopping:
                # Get the metric to monitor
                metric_name = early_stopping_params.get("metric", "loss")
                current_metric = val_metrics[metric_name]

                # Call early stopping with current metric
                model_path = f"runs/{model_name}_early_stopping.pth"
                if early_stopping(epoch, current_metric, model, model_path):
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(
                        f"Best {metric_name} was at epoch {early_stopping.best_epoch}"
                    )
                    break

        # Log embeddings after training is complete
        log_embeddings(writer, model, val_loader, device, class_names, epoch)

        # Log final hyperparameters with metrics
        final_metrics = {
            "hparam/val_accuracy": val_metrics["accuracy"],
            "hparam/val_f1": val_metrics["f1"],
            "hparam/val_loss": val_metrics["loss"],
            "hparam/epochs_trained": epoch,
        }
        writer.add_hparams(hyperparams, final_metrics)

    # Set the model to evaluation mode after training
    model.eval()
    print(f"Training complete. Best validation F1: {best_val_f1:.4f}")

    # If early stopping was used, report the best epoch
    if early_stopping:
        print(
            f"Best {early_stopping_params.get('metric', 'loss')} was at epoch {early_stopping.best_epoch}"
        )

    return best_metrics
