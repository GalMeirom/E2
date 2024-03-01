import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def syntheticData(num_samples, seq_length, input_size):
    input_sequences = torch.rand(num_samples, seq_length, input_size)

    # Step 2: Post-process each sequence
    for sequence in input_sequences:
        i = torch.randint(20, 30, size=(1,))  # Sample i from [20, 30]
        start_index = max(0, i - 5)
        end_index = min(seq_length, i + 6)
        sequence[start_index:end_index] *= 0.1
    return input_sequences


def mnist():
    # Download MNIST dataset with pytorch
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    
    # Define a transformation to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    return trainloader, testloader

mnist()
