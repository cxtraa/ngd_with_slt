import torch as t
from torch import nn
import torch.nn.functional as F
    
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
    
class LeNet(nn.Module):
    def __init__(self,dataset, extra_layers=0, output_dim=10):
        super().__init__()
        if dataset=='mnist':
            in_channels=1
            fc_1=16*4*4
        elif dataset=='cifar10':
            in_channels=3
            fc_1=16*5*5

        self.conv1 =  nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1),
                nn.ReLU()
        )
        
        self.extra= nn.ModuleList([
            nn.Sequential(
                #padding of 2 here keeps the size the same at [batch_size, 6, 24, 24]
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5, stride=1, padding=2),
                nn.ReLU()
            )
            for _ in range(extra_layers)
        ])

        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                nn.ReLU()
        )

        self.fc_1 = nn.Linear(fc_1, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):

        # x = [batch size, 1, 28, 28]

        x = self.conv1(x)
        # x = [batch size, 6, 24, 24]

        for layer in self.extra:
            x = layer(x)  # Apply each layer in the ModuleList

        x = F.max_pool2d(x, kernel_size=2)
        # x = [batch size, 6, 12, 12]

        x = self.conv2(x)
        # x = [batch size, 16, 8, 8]

        x = F.max_pool2d(x, kernel_size=2)
        # x = [batch size, 16, 4, 4]

        x = t.flatten(x, 1)

        x = self.fc_1(x)
        # x = [batch size, 120]
        x = F.relu(x)

        x = self.fc_2(x)
        # x = batch size, 84]
        x = F.relu(x)

        x = self.fc_3(x)
        # x = [batch size, output dim]

        return x
    
class NeuralNet(nn.Module):
    """
    Simple Neural Network
    relu: input False for deep linear neural network, and true for feed forward neural network

    outputsize should be 10 for MNIST (num classes)
    """
    
    def __init__(self, relu, input_size, hidden_layers=1, hidden_nodes=16, output_size=10):
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