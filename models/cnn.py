import torch
from torch import nn


class CNN(nn.Module):
    """ Convolutional Neural Network for multiclass classification.

    This CNN follows the architecture of LeNet. However, instead of using
    Tanh activation function and average pooling, ReLU activation function
    and max pooling is used. Also, dropout layers and batch normalization layers have been added.

    The CNN architecture is as follows:
    Input → CONV1 → NORM1 → ReLU → POOL2 → CONV3 → NORM2 → ReLU → POOL4 → CONV5 → FC6 → Softmax

    The convolutional layers are configured as follows:
        * CONV1: 6 filters and kernel size of 5 x 5
        * CONV3: 16 filters and kernel size of 5 x 5
        * CONV5: 120 filters and kernel size of 5 x 5

    The pooling layers are configured as follwos:
        * POOL2, POOL4: max pooling layers with receptive field of 2 x 2.

    The hyperparameters of the CNN are:
        * probability of dropout layer.

    Arguments:
        in_channels:
            size of input, as an integer.
        output_size:
            size of output (number of labels), as an integer
        p:
            dropout probability, as a float.
    """

    def __init__(self, in_channels, output_size, p):
        """ Initialize Convolutional Neural Network.

        Args:
            in_channels:
                size of input, as an integer.
            output_size:
                size of output (number of labels), as an integer.
            p:
                dropout probability, as a float.
        """
        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.output_size = output_size
        self.p = p

        # activation function
        self.activation = nn.ReLU()

        # convolutional layer: CONV1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1)
        # convolutional layer: CONV3
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # dropout layer
        self.dropout = nn.Dropout(p)

        # batch normalization layer: NORM1
        self.norm1 = nn.BatchNorm2d(6)
        # batch normalization layer: NORM2
        self.norm2 = nn.BatchNorm2d(16)

        # fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=output_size)

    def forward(self, x):
        # Input --> CONV1 --> NORM --> ReLU --> POOL2
        pre = self.conv1(x)
        normalized = self.norm1(pre)
        post = self.activation(normalized)
        x = self.pool(post)

        # CONV3 → NORM → ReLU → POOL4
        pre = self.conv2(x)
        normalized = self.norm2(pre)
        post = self.activation(normalized)
        x = self.pool(post)

        # flatten all dimensions except batch
        x = torch.flatten(x, 1)

        # CONV5
        logits = self.activation(self.fc1(x))
        logits = self.dropout(logits)

        # FC6
        logits = self.activation(self.fc2(logits))
        logits = self.dropout(logits)

        # outpout
        logits = self.fc3(logits)

        return logits
