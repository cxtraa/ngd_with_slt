
import torch.nn as nn
import torch.nn.functional as F
import torch as t

class LeNet(nn.Module):
    def __init__(self, extra_layers=0, output_dim=10):
        super().__init__()

        self.conv1 =  nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
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

        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
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