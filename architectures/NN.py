from torch import nn

class NeuralNet(nn.Module):
    """
    Simple Neural Network
    relu: input False for deep linear neural network, and true for feed forward neural network

    outputsize should be 10 for MNIST (num classes)
    """
    
    def __init__(self, relu, input_size=28*28, hidden_layers=1, hidden_nodes=16, output_size=10):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes

        layers = [nn.Flatten()]
        for i in range(hidden_layers):
            linear_layer = nn.Linear(input_size if i == 0 else hidden_nodes, hidden_nodes)
            layers.append(linear_layer)

            if relu:
                layers.append(nn.ReLU())

        output_layer = nn.Linear(hidden_nodes, output_size)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)