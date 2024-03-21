import torch as t
from torch import nn

class LinearMNIST(nn.Module):
    """
    Simple template NN architecture.
    Adjust architecture accordingly for experiment.
    """
    def __init__(self, input_size=28*28, hidden_layers=1, hidden_nodes=16, output_size=10):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes

        layers = [nn.Flatten()]
        for i in range(hidden_layers):
            linear_layer = nn.Linear(input_size if i == 0 else hidden_nodes, hidden_nodes)
            layers.append(linear_layer)
            layers.append(nn.ReLU())
        output_layer = nn.Linear(hidden_nodes, output_size)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CnnMNIST(nn.Module):
    """
    CNN architecture for training on MNIST.
    output_size : number of classification categories
    kernel_size : size of kernel for convolution
    hidden_conv_layers : number of hidden convolutional layers in the network
    """
    def __init__(self, output_size=10, kernel_size=3, hidden_conv_layers=1):
        super().__init__()
        self.conv_layers = []
        for _ in range(hidden_conv_layers):
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                nn.ReLU(),
            )
            self.conv_layers.append(conv_layer)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            *self.conv_layers,
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_size)
        )
    
    def forward(self, x):
        return self.model(x)
    
class LinearCIFAR10(nn.Module):
    """
    Simple template NN architecture for CIFAR10.
    Adjust architecture accordingly for experiment.
    """
    def __init__(self, input_size=32*32*3, hidden_layers=1, hidden_nodes=16, output_size=10):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes

        layers = [nn.Flatten()]
        for i in range(hidden_layers):
            linear_layer = nn.Linear(input_size if i == 0 else hidden_nodes, hidden_nodes)
            layers.append(linear_layer)
            layers.append(nn.ReLU())
        output_layer = nn.Linear(hidden_nodes, output_size)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class CnnCIFAR10(nn.Module):
    """
    CNN architecture for training on CIFAR10.
    output_size : number of classification categories
    kernel_size : size of kernel for convolution
    hidden_conv_layers : number of hidden convolutional layers in the network
    """
    def __init__(self, output_size=10, kernel_size=3, hidden_conv_layers=1):
        super().__init__()
        self.conv_layers = []
        for _ in range(hidden_conv_layers):
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                nn.ReLU(),
            )
            self.conv_layers.append(conv_layer)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            *self.conv_layers,
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_size)
        )
    
    def forward(self, x):
        return self.model(x)
    
class MinimallySingularModel(nn.Module):
    """
    An artifically constructed model that has d parameters, of which only
    r are used, where r < d.

    The model output is the sum of squares of the first r parameters, making
    the model minimally singular.
    """
    def __init__(self, params, r):
        super().__init__()
        d = params.shape[0]        
        assert r <= d, "r must be less than or equal to d."

        self.r = r
        self.params = nn.Parameter(params, requires_grad=True)
    
    def forward(self, x):
        return x * t.sum(self.params[:self.r]**2)