import datetime
import os

import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T


class FashionMNIST:
    def __init__(self, root: str = "./data", batch_size=64, resize=(28, 28)) -> None:
        self.root = root
        self.batch_size = batch_size
        transformations = T.Compose([T.Resize(resize), T.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=transformations, download=True
        )
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=transformations, download=True
        )

    def get_dataloader(self, train: bool, num_workers: int = 0) -> DataLoader:
        """
        The DataLoader allows us to iterate over the dataset in batches.
        """
        dataset = self.train if train else self.val
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=train, num_workers=num_workers
        )


class SoftmaxRegression(nn.Module):
    def __init__(self, num_outputs: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(784, num_outputs)
        self.lazy_linear = nn.LazyLinear(num_outputs)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.flatten(x)
        out = self.linear(out)
        out = self.log_softmax(out)
        return out


def train(n_epochs, optimizer, model, loss_fn, train_loader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(f"Training on {device}")

    for epoch in range(n_epochs):
        train_loss = 0.0
        for images, labels in train_loader:
            # Move the images and labels to the device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the training loss
            train_loss += loss.item()

        if epoch % 10 == 0:
            print(
                f"{datetime.datetime.now()} - Epoch {epoch} - Training loss: {train_loss / len(train_loader)}"
            )


def validate(model, train_loader, val_loader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(f"Validation on {device}")

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy {name}: {correct / total}")


if __name__ == "__main__":
    learning_rate = 1e-2

    model = SoftmaxRegression(num_outputs=10)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = F.cross_entropy

    data_loader = FashionMNIST(batch_size=2048)

    num_workers = os.cpu_count()
    train_loader = data_loader.get_dataloader(train=True, num_workers=num_workers)
    val_loader = data_loader.get_dataloader(train=False, num_workers=num_workers)

    train(
        n_epochs=100,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
    )

    validate(model, train_loader, val_loader)
