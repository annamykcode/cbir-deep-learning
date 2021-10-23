import torch
from torch import nn


class LeNet5(nn.Module):
    """ Represents LeNet-5 convolutional neural network. """

    def __init__(self, output_size):
        """ Initialize LeNet-5.

        Args:
            output_size:
                size of output (number of labels), as an integer.
        """
        super(LeNet5, self).__init__()

        # activation function
        self.activation = nn.Tanh()

        # convolutional layer: CONV1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)

        # convolutional layer: CONV3
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # average pooling layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=output_size)

    def forward(self, x):
        # Input --> CONV1 --> NORM --> ReLU --> POOL2
        x = self.pool(self.activation(self.conv1(x)))

        # CONV3 → NORM → ReLU → POOL4
        x = self.pool(self.activation(self.conv2(x)))

        # flatten all dimensions except batch
        x = torch.flatten(x, 1)

        # CONV5
        logits = self.activation(self.fc1(x))

        # FC6
        logits = self.activation(self.fc2(logits))

        # outpout
        logits = self.fc3(logits)

        return logits
