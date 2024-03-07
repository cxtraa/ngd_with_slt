# from torch import nn

# class CnnMNIST(nn.Module):
#     def __init__(self, output_size=10, kernel_size=3, hidden_conv_layers=1):
#         super().__init__()
#         self.conv_layers = []
#         for _ in range(hidden_conv_layers):
#             conv_layer = nn.Sequential(
#                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
#                 nn.ReLU(),
#                 # nn.MaxPool2d(kernel_size=2, stride=2),
#             )
#             self.conv_layers.append(conv_layer)

#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             *self.conv_layers,
            
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Flatten(),
#             nn.Linear(in_features=64 * 7 * 7, out_features=128),
#             nn.ReLU(),
#             nn.Linear(in_features=128, out_features=output_size)
#         )
    
#     def forward(self, x):
#         return self.model(x)
    
#     def forward(self, x):
#     print(f"Input shape: {x.shape}")

#     for i, layer in enumerate(self.model):
#         x = layer(x)
#         print(f"Shape after layer {i} ({layer.__class__.__name__}): {x.shape}")
        
from torch import nn
import torch.nn.functional as F
import torch as t

class CnnMNIST(nn.Module):
    def __init__(self, output_size=10, kernel_size=3, hidden_conv_layers=1):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size, stride=1, padding=kernel_size//2)
        self.conv_hidden = nn.ModuleList([
            nn.Conv2d(32, 32, kernel_size, stride=1, padding=kernel_size//2) 
            for _ in range(hidden_conv_layers)
        ])
        self.conv_final = nn.Conv2d(32, 64, kernel_size, stride=1, padding=kernel_size//2)
        
        # Placeholders for fully connected layers. We'll initialize them after computing the flattened size
        #note that 64*7*7 is the wrong calculation, actual empirical input size is 5184
        self.fc1 = nn.Linear(5184, 128)
        self.fc2 = nn.Linear(128, output_size)

        # Output size placeholder
        self.output_size = output_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        for conv in self.conv_hidden:
            x = F.relu(conv(x))

        x = F.relu(self.conv_final(x))
        x = F.max_pool2d(x, 2)

        x = t.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    


