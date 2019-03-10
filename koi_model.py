"""Code for a small neural network architecture."""
from torch.nn import Module, Conv1d, Linear
from torch.nn.functional import leaky_relu, max_pool1d

from koi_dataset import padded_example_length


class SmallNet(Module):
    """A small network for the KOI experiment."""
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1d(1, 16, kernel_size=3)
        self.conv2 = Conv1d(self.conv1.out_channels, 16, kernel_size=3)
        self.conv3 = Conv1d(self.conv2.out_channels, 16, kernel_size=3)
        self.conv4 = Conv1d(self.conv3.out_channels, 16, kernel_size=3)
        self.conv5 = Conv1d(self.conv4.out_channels, 16, kernel_size=3)
        self.conv6 = Conv1d(self.conv6.out_channels, 16, kernel_size=3)
        self.linear = Linear(20, 1)

    def forward(self, x):
        """The forward pass of the network."""
        x = leaky_relu(self.conv1(x))
        x = max_pool1d(x, kernel_size=2)
        x = leaky_relu(self.conv2(x))
        x = max_pool1d(x, kernel_size=2)
        x = leaky_relu(self.conv3(x))
        x = max_pool1d(x, kernel_size=3)
        x = leaky_relu(self.conv4(x))
        x = max_pool1d(x, kernel_size=3)
        x = leaky_relu(self.conv5(x))
        x = max_pool1d(x, kernel_size=3)
        x = leaky_relu(self.conv6(x))
        x = max_pool1d(x, kernel_size=3)
        x = x.view(-1, 20)
        x = self.linear(x)
        x = x.view(-1)
        return x
