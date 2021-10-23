from torch import nn


class NeuralNetwork(nn.Module):
    """ Neural network for multiclass classification.

    This neural network uses a ReLU activation function in hidden layers and a Softmax activation
    function in output layer. Also, dropout layers have been defined.
    The hyperparameters of the neural network are:
        * number of hidden layers,
        * size of hidden layers (number of neurons),
        * probability of dropout layer.

    Arguments:
        input_size:
            size of input, as an integer.
        hidden_size:
            number of neurons in hidden layers, as an integer.
        output_size:
            size of output (number of labels), as an integer.
        num_layers:
            number of hidden layers, as an integer.
        p:
            dropout probability, as a float.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, p):
        """ Initialize neural network.

        Args:
            input_size:
                size of input, as an integer.
            hidden_size:
                number of neurons in hidden layers, as an integer.
            output_size:
                size of output (number of labels), as an integer.
            num_layers:
                number of hidden layers, as an integer.
            p:
                dropout probability, as a float.
        """
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.p = p

        # activation function
        self.activation = nn.ReLU()

        # flatten layer
        self.flatten = nn.Flatten()

        # dropout layer
        self.dropout = nn.Dropout(p)

        # input layer
        self.input_linear = nn.Linear(input_size, hidden_size)

        # hidden layers
        self.hidden_linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])

        # output layer
        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # flatten input
        logits = self.flatten(x)

        # input layer
        pre = self.input_linear(logits)
        post = self.activation(pre)
        logits = self.dropout(post)

        # hidden layers
        for i, layer in enumerate(self.hidden_linears):
            pre = layer(logits)
            post = self.activation(pre)
            logits = self.dropout(post)

        # output layer
        logits = self.output_linear(logits)

        return logits
