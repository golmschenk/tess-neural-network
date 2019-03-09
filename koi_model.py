"""Code for a small neural network architecture."""
from torch.nn import Module


class SmallNet(Module):
    """A small network for the KOI experiment."""
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        """The forward pass of the network."""
        return x
