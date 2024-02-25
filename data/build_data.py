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
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    except Exception as e:
        print('Cannot find data')

    train_loader = t.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,persistent_workers=True)
    test_loader = t.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,persistent_workers=True)

    return train_loader, test_loader
