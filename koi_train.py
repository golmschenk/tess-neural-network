"""Code for training to identify KOI."""
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from koi_dataset import KoiCatalogDataset
from koi_model import SimplePoolingConvolutionalNet, SimpleStridedConvolutionalNet


def train():
    """Trains and evaluates the KOI network."""
    batch_size = 100
    validation_dataset_size = 2000
    train_dataset = KoiCatalogDataset(start=None, end=-validation_dataset_size)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_dataset = KoiCatalogDataset(start=-validation_dataset_size, end=None)
    validation_dataset_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    network = SimpleStridedConvolutionalNet()
    optimizer = Adam(network.parameters())
    criterion = BCEWithLogitsLoss()

    for epoch in range(300):
        total_train_correct = 0
        for examples, labels in train_dataset_loader:
            optimizer.zero_grad()
            predicted_scores = network(examples)
            loss = criterion(predicted_scores, labels)
            loss.backward()
            optimizer.step()
            predicted_labels = torch.sigmoid(predicted_scores.detach()).round().numpy()
            total_train_correct += np.sum(predicted_labels == labels.detach().numpy())
        train_accuracy = total_train_correct / len(train_dataset)
        with torch.no_grad():  # For speed, don't calculate gradient during evaluation.
            total_validation_correct = 0
            for validation_examples, validation_labels in validation_dataset_loader:
                predicted_validation_scores = network(validation_examples)
                predicted_validation_labels = torch.sigmoid(predicted_validation_scores.detach()).round().numpy()
                total_validation_correct += np.sum(predicted_validation_labels == validation_labels.detach().numpy())
            validation_accuracy = total_validation_correct / len(validation_dataset)
        print(f'Epoch: {epoch}, Train accuracy: {train_accuracy:.5f}, Validation accuracy: {validation_accuracy:.5f}')


if __name__ == '__main__':
    train()
