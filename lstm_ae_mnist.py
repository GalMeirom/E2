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
    torch.manual_seed(42)
    train_dataloader, testloader = Data.mnist()
    seq_length = 28
    input_size = 28
    
    epochs = 15
    models_dict = {}

    param_grid = {
    'input_size': [input_size],
    'num_layers': [1],
    'hidden_size': [64],
    'learning_rate': [0.01],
    'clip_value' : [1.0],
    'seq_length': [seq_length],
    'numC': [1],
    }

    models_dict = {}
    for params in ParameterGrid(param_grid):
        print(str(params))
        model = lstm.LSTM_Model(**params)
        model.train(train_dataloader, epochs)
        #models_dict[str(params)] = [model, model.eval(testloader)]
        ut.reImage(model, testloader)
        print()


def Q1C2():
    torch.manual_seed(42)
    train_dataloader, testloader = Data.mnist()
    seq_length = 28
    input_size = 28
    
    epochs = 20

    param_grid = {
    'input_size': [input_size],
    'num_layers': [1],
    'hidden_size': [32],
    'learning_rate': [0.001],
    'clip_value' : [1.0],
    'seq_length': [seq_length],
    'numC': [10],
    }

    accuracies = []
    label_losses = []
    for params in ParameterGrid(param_grid):
        model = lstm.LSTM_Model_wPred(**params)
        for i in range(epochs):
            avg_label_train_loss, avg_rec_train_loss = model.train(train_dataloader)
            print(f'Epoch [{i+1}/{epochs}], AVG Label Training Loss: {avg_label_train_loss}')
            print(f'Epoch [{i+1}/{epochs}], AVG Reconstruct Training Loss: {avg_rec_train_loss}')
            accuracy, label_loss = model.pred(testloader)
            print(accuracy)
            ut.reImage(model, testloader)
            accuracies.append(accuracy)
            label_losses.append(label_loss)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))

    # Plot the first Y array on the first subplot
    ax1.plot(range(1, epochs+1), accuracies, label='Accuracy', color='blue')
    ax1.set_xlabel('Ephocs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs ephocs')
    ax1.legend()

    # Plot the second Y array on the second subplot
    ax2.plot(range(1, epochs+1), label_losses, label='CE Loss', color='red')
    ax2.set_xlabel('Ephocs')
    ax2.set_ylabel('CE Loss')
    ax2.set_title('Average CE Loss vs ephocs')
    ax2.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()







Q1C2()