from torch import nn

class CnnMNIST(nn.Module):
    def __init__(self, output_size=10, kernel_size=3, hidden_conv_layers=1):
        super().__init__()
        self.conv_layers = []
        for _ in range(hidden_conv_layers):
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2, stride=2),
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