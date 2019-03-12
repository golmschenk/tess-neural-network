"""Code for a simple convolutional neural network architecture."""
import torch
from torch.nn import Module, Conv1d, Linear, BatchNorm1d
from torch.nn.functional import leaky_relu, max_pool1d, avg_pool1d

from koi_dataset import padded_example_length


class SimplePoolingConvolutionalNet(Module):
    """A simple convolutional network for the KOI experiment."""
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1d(1, 32, kernel_size=3)
        self.conv2 = Conv1d(self.conv1.out_channels, 64, kernel_size=3)
        self.conv3 = Conv1d(self.conv2.out_channels, 128, kernel_size=3)
        self.conv4 = Conv1d(self.conv3.out_channels, 128, kernel_size=3)
        self.conv5 = Conv1d(self.conv4.out_channels, 64, kernel_size=3)
        self.conv6 = Conv1d(self.conv5.out_channels, 64, kernel_size=3)
        self.linear1 = Linear(self.conv6.out_channels, 16)
        self.linear2 = Linear(self.linear1.out_features, 1)

    def forward(self, x):
        """The forward pass of the network."""
        x = x.view(-1, 1, padded_example_length)
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
        x = avg_pool1d(x, kernel_size=459)
        x = x.view(-1, self.conv6.out_channels)
        x = leaky_relu(self.linear1(x))
        x = self.linear2(x)
        x = x.view(-1)
        return x


class SimpleStridedConvolutionalNet(Module):
    """A simple convolutional network for the KOI experiment."""
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1d(1, 8, kernel_size=3, stride=2)
        self.conv2 = Conv1d(self.conv1.out_channels, 16, kernel_size=3, stride=2)
        self.conv3 = Conv1d(self.conv2.out_channels, 16, kernel_size=3, stride=3)
        self.bn3 = BatchNorm1d(self.conv3.out_channels)
        self.conv4 = Conv1d(self.conv3.out_channels, 32, kernel_size=3, stride=3)
        self.bn4 = BatchNorm1d(self.conv4.out_channels)
        self.conv5 = Conv1d(self.conv4.out_channels, 32, kernel_size=3, stride=3)
        self.bn5 = BatchNorm1d(self.conv5.out_channels)
        self.conv6 = Conv1d(self.conv5.out_channels, 64, kernel_size=3, stride=3)
        self.bn6 = BatchNorm1d(self.conv6.out_channels)
        self.conv7 = Conv1d(self.conv6.out_channels, 64, kernel_size=3, stride=3)
        self.bn7 = BatchNorm1d(self.conv7.out_channels)
        self.conv8 = Conv1d(self.conv7.out_channels, 128, kernel_size=3, stride=3)
        self.bn8 = BatchNorm1d(self.conv8.out_channels)
        self.conv9 = Conv1d(self.conv8.out_channels, 128, kernel_size=3, stride=3)
        self.bn9 = BatchNorm1d(self.conv9.out_channels)
        self.conv10 = Conv1d(self.conv9.out_channels, 8, kernel_size=1)
        self.linear1 = Linear(40, 8)
        self.linear2 = Linear(self.linear1.out_features, 1)

    def forward(self, x):
        """The forward pass of the network."""
        x = x.view(-1, 1, padded_example_length)
        x = leaky_relu(self.conv1(x))
        x = leaky_relu(self.conv2(x))
        x = leaky_relu(self.conv3(x))
        x = self.bn3(x)
        x = leaky_relu(self.conv4(x))
        x = self.bn4(x)
        x = leaky_relu(self.conv5(x))
        x = self.bn5(x)
        x = leaky_relu(self.conv6(x))
        x = self.bn6(x)
        x = leaky_relu(self.conv7(x))
        x = self.bn7(x)
        x = leaky_relu(self.conv8(x))
        x = self.bn8(x)
        x = leaky_relu(self.conv9(x))
        x = self.bn9(x)
        x = leaky_relu(self.conv10(x))
        x = x.view(-1, self.linear1.in_features)
        x = leaky_relu(self.linear1(x))
        x = self.linear2(x)
        x = x.view(-1)
        return x
