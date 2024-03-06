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
    # Step 1: Generate 10,000 input sequences of length 50 with values in the range [0, 1] as PyTorch tensors
    num_samples = 10000
    seq_length = 50
    input_sequences = Data.syntheticData(num_samples, seq_length)

    # Step 3: Split the data
    X_train_tensor, X_temp_tensor = train_test_split(input_sequences, test_size=0.4, random_state=18)
    X_val_tensor, X_test_tensor = train_test_split(X_temp_tensor, test_size=0.5, random_state=18)


    # # Step 4: Print some example sequences
    print("Example sequences from training set:")
    for i in range(3):
        print("Sequence", i+1, ":", X_train_tensor[i])

    # # Step 5: Plot some example sequences
    plt.figure(figsize=(10, 6))

    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(X_train_tensor[i], label='Sequence {}'.format(i+1))
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('Training Set - Sequence {}'.format(i+1))
        plt.legend()

    plt.tight_layout()
    plt.show()

def Q1C2():
    torch.manual_seed(42)

    num_of_seq = 10000 
    seq_length = 50
    input_size = 1
    data = Data.syntheticData(num_of_seq, seq_length, input_size)  # 100 sequences, each of length 20
    
    train_tensor, temp_tensor = train_test_split(data, test_size=0.4, random_state=18)
    val_tensor, test_tensor = train_test_split(temp_tensor, test_size=0.5, random_state=18)
    
    train_dataset = TensorDataset(train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)    
    
    val_dataset = TensorDataset(val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)    

    epochs = 40
    models_dict = {}

    param_grid = {
    'input_size': [input_size],
    'num_layers': [1],
    'hidden_size': [32, 40, 48],
    'learning_rate': [0.01, 0.05, 0.075],
    'clip_value' : [1.0, 1.5, 2.0],
    'seq_length': [seq_length],
    'numC': [1],
    'apprx': [0],
    }
    
    best_params = None
    models_dict = {}
    for params in ParameterGrid(param_grid):
        print(str(params))
        model = lstm.LSTM_Model(**params)
        model.model.double()
        model.train(train_dataloader, epochs)
        models_dict[str(params)] = [model, model.eval(val_dataloader)]
        print(f'{str(params)} - {models_dict[str(params)][1]}')
        print(params['input_size'])
        ut.reconstruct(models_dict[str(params)][0], val_tensor)
    
    best_params = min(models_dict, key=lambda k: models_dict[k][1])
    best_model = models_dict[best_params]

    print("Best Hyperparameters:", best_params)
    print("Best Validation Loss:", best_model[1])

Q1C2()