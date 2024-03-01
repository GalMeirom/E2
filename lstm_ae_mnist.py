import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import Data
import utils as ut
import lstm

def Q1C1():
    num_of_seq = 5000 
    seq_length = 28
    input_size = 28
    trainloader, testloader = Data.mnist()
    print(len(trainloader))
    print('Hello World')

Q1C1()