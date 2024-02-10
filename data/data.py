'''
Builds train and test dataLoader objects
'''
import torch as t
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
def build_data(args):
    # Load MNIST data

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = t.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = t.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader