import datetime
from contextlib import contextmanager

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter


def train_epoch(model, train_loader, optimizer, loss_fn, device, clipping=None):
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

    for X, y in train_loader:
        X = X.to(device=device)
        y = y.to(device=device)

        # Forward pass
        outputs = model(X)

        # Handle different output formats
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Get the main output if it's a tuple

        # For sequence models, reshape if needed
        if outputs.dim() > 2:
            # If outputs is [batch, seq_len, vocab_size], transpose to [batch, vocab_size, seq_len]
            # for CrossEntropyLoss which expects [N, C, d1, d2, ...]
            outputs = (
                outputs.transpose(1, 2)
                if outputs.shape[1] != outputs.shape[2]
                else outputs
            )

        loss = loss_fn(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        if clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
        optimizer.step()

        loss_train += loss.item()

        # Calculate accuracy
        if outputs.dim() <= 2:
            # Standard classification
            _, predicted = torch.max(outputs, dim=1)
        else:
            # For sequence prediction, get prediction at each position
            _, predicted = torch.max(outputs, dim=1)

        total_train += y.numel()  # Count all elements in y
        correct_train += (predicted == y).sum().item()

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
        for X, y in val_loader:
            X = X.to(device=device)
            y = y.to(device=device)

            # For sequence models, the output shape might be different
            outputs = model(X)

            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get the main output if it's a tuple

            # For sequence models, reshape if needed
            if outputs.dim() > 2:
                # If outputs is [batch, seq_len, vocab_size], transpose to [batch, vocab_size, seq_len]
                # for CrossEntropyLoss which expects [N, C, d1, d2, ...]
                outputs = (
                    outputs.transpose(1, 2)
                    if outputs.shape[1] != outputs.shape[2]
                    else outputs
                )

            loss = loss_fn(outputs, y)
            val_loss += loss.item()

            # Get predictions
            if outputs.dim() <= 2:
                # Standard classification
                _, predicted = torch.max(outputs, dim=1)
            else:
                # For sequence prediction, get prediction at each position
                _, predicted = torch.max(outputs, dim=1)

            total_val += y.numel()  # Count all elements in y
            correct_val += (predicted == y).sum().item()

            # Flatten predictions and labels for metrics calculation
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(y.view(-1).cpu().numpy())

    # Calculate validation metrics
    val_loss = val_loss / len(val_loader)
    val_accuracy = correct_val / total_val

    # Calculate F1 score if we have valid labels
    unique_labels = np.unique(all_labels + all_preds)
    val_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        labels=unique_labels if len(unique_labels) > 1 else None,
    )

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
        try:
            # Get a batch of data
            example_inputs, _ = next(iter(train_loader))
            example_inputs = example_inputs.to(device)

            # Try to add the graph
            writer.add_graph(model, example_inputs)
        except Exception as e:
            print(f"Failed to add model graph to TensorBoard: {e}")
            print("Continuing without model graph visualization.")

    # Log histograms of model parameters
    for name, param in model.named_parameters():
        writer.add_histogram(f"Parameters/{name}", param, epoch)
        if param.grad is not None:
            writer.add_histogram(f"Gradients/{name}", param.grad, epoch)


def log_predictions(
    writer, model, val_loader, device, class_names, epoch, num_samples=5
):
    """Log prediction visualizations to TensorBoard for text data.

    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to run on
        class_names (list): List of class names (vocabulary)
        epoch (int): Current epoch
        num_samples (int): Number of text samples to visualize
    """
    model.eval()
    samples_so_far = 0
    prediction_text = ""

    with torch.no_grad():
        for X, y in val_loader:
            if samples_so_far >= num_samples:
                break

            X = X.to(device)
            y = y.to(device)

            # Get model outputs
            outputs = model(X)

            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Get predictions
            if outputs.dim() <= 2:
                _, preds = torch.max(outputs, dim=1)
            else:
                _, preds = torch.max(outputs, dim=2)  # For sequence models

            # For each sample in the batch
            for j in range(min(X.size(0), num_samples - samples_so_far)):
                # Convert input sequence to text
                input_text = "Input: "
                for idx in X[j].cpu().numpy():
                    if idx < len(class_names):
                        input_text += class_names[idx]

                # Convert true labels to text
                true_text = "True: "
                for idx in y[j].cpu().numpy():
                    if idx < len(class_names):
                        true_text += class_names[idx]

                # Convert predictions to text
                pred_text = "Pred: "
                for idx in preds[j].cpu().numpy():
                    if idx < len(class_names):
                        pred_text += class_names[idx]

                # Add to overall prediction text
                prediction_text += f"Sample {samples_so_far + 1}:\n{input_text}\n{true_text}\n{pred_text}\n\n"
                samples_so_far += 1

                if samples_so_far >= num_samples:
                    break

    # Log the text predictions
    writer.add_text(f"Predictions/Epoch_{epoch}", prediction_text, epoch)


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
    # For RNN models, we'll skip embedding visualization or implement a custom approach
    try:
        # Collect some input data
        all_inputs = []
        all_labels = []

        # Limit the number of samples for visualization
        max_samples = 1000
        samples_collected = 0

        for X, y in val_loader:
            if samples_collected >= max_samples:
                break

            batch_size = X.size(0)
            if samples_collected + batch_size > max_samples:
                # Take only what we need to reach max_samples
                X = X[: max_samples - samples_collected]
                y = y[: max_samples - samples_collected]

            all_inputs.append(X)
            all_labels.append(y)
            samples_collected += X.size(0)

        if all_inputs:
            # Concatenate all collected data
            all_inputs = torch.cat(all_inputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Get embeddings - this is model specific and might need adjustment
            with torch.no_grad():
                # For RNN models, we might want to get the hidden state
                # This is just a placeholder - adapt to your specific model
                all_inputs = all_inputs.to(device)
                if hasattr(model, "get_embeddings"):
                    embeddings = model.get_embeddings(all_inputs)
                else:
                    # Try to get the first hidden state as embedding
                    outputs, hidden = model.rnn(model.one_hot(all_inputs))
                    embeddings = hidden.cpu().numpy()

            # Create metadata labels
            metadata = [f"Sample_{i}" for i in range(len(embeddings))]

            # Log the embeddings
            writer.add_embedding(embeddings, metadata=metadata, global_step=n_epochs)

    except Exception as e:
        print(f"Failed to log embeddings to TensorBoard: {e}")
        print("Continuing without embedding visualization.")


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
    device,
    class_names,
    model_name="model",
    early_stopping_params=None,
    clipping=None,
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
            train_metrics = train_epoch(
                model, train_loader, optimizer, loss_fn, device, clipping
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
