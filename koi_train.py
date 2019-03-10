"""Code for training to identify KOI."""
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from koi_dataset import KoiCatalogDataset
from koi_model import SmallNet


def train():
    """Trains and evaluates the KOI network."""
    batch_size = 100
    validation_dataset_size = 2000
    train_dataset = KoiCatalogDataset(start=None, end=-validation_dataset_size)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = KoiCatalogDataset(start=-validation_dataset_size, end=None)
    validation_dataset_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    network = SmallNet()
    optimizer = Adam(network.parameters())
    criterion = BCEWithLogitsLoss()

    for epoch in range(300):
        total_train_incorrect = 0
        for examples, labels in train_dataset_loader:
            optimizer.zero_grad()
            predicted_scores = network(examples)
            loss = criterion(predicted_scores, labels)
            loss.backward()
            optimizer.step()
            predicted_labels = torch.sigmoid(predicted_scores.detach()).round().numpy()
            total_train_incorrect += np.sum(predicted_labels != labels.detach().numpy())
        train_error = total_train_incorrect / len(train_dataset)
        with torch.no_grad():  # For speed, don't calculate gradient during evaluation.
            total_validation_incorrect = 0
            for validation_examples, validation_labels in validation_dataset_loader:
                predicted_validation_scores = network(validation_examples)
                predicted_validation_labels = torch.sigmoid(predicted_validation_scores.detach()).round().numpy()
                total_validation_incorrect += np.sum(predicted_validation_labels != validation_labels.detach().numpy())
            validation_error = total_validation_incorrect / len(validation_dataset)
        print('Epoch: {}, Train error: {:.5f}, Validation error: {:.5f}'.format(epoch, train_error, validation_error))


if __name__ == '__main__':
    train()
