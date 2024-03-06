import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd

def syntheticData(num_samples, seq_length, input_size):
    input_sequences = torch.rand(num_samples, seq_length, input_size)

    # Step 2: Post-process each sequence
    for sequence in input_sequences:
        i = torch.randint(20, 30, size=(1,))  # Sample i from [20, 30]
        start_index = max(0, i - 5)
        end_index = min(seq_length, i + 6)
        sequence[start_index:end_index] *= 0.1
    return input_sequences.double()


def mnist():
    # Download MNIST dataset with pytorch
    
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



def reshaped_mnist(rows,cols):
    transform = transforms.Compose([
        transforms.Resize((rows, cols)),  # Resize the image to the specified dimensions
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=True, transform=transform)
  
    # Flatten the dataset
    flattened_data = []
    for image, label in trainset:
        flattened_data.append(image.flatten().unsqueeze(0))
    # Concatenate the flattened images along the batch dimension
    flattened_data = torch.cat(flattened_data, dim=0)
    # Reshape the tensor to have a batch size of 1 and -1 in the second dimension
    trainset.data = flattened_data
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        
    # Download and load the test data
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=False, transform=transform)
    flattened_data = []
    for image, label in testset:
        flattened_data.append(image.flatten().unsqueeze(0))
    flattened_data = torch.cat(flattened_data, dim=0)
    testset.data = flattened_data
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    return trainloader, testloader


def snp500(seq_length):
    df = pd.read_csv('Data\SP 500 Stock Prices 2014-2017.csv', header=None)
    df.columns = df.iloc[0]

    # Drop the first row
    df = df[1:].reset_index(drop=True)
    data = df.dropna()
    data['date'] = pd.to_datetime(data['date'])
    data['t0'] = (data['date'] - data['date'].min()).dt.days.astype(int)
    data = data.drop(['volume', 'date', 'open', 'close', 'low'], axis= 1)
    data['high'] = pd.to_numeric(data['high'])
    df_sorted = data.sort_values(by=['symbol', 't0'])

    # Define a function to extract consecutive chunks of 50 rows and drop 'symbol' and 't0'
    def extract_consecutive_chunks(group):
        chunks = [group.iloc[i:i+seq_length].reset_index(drop=True).drop(['symbol', 't0'], axis=1) for i in range(0, len(group), seq_length) if len(group[i:i+seq_length]) == seq_length]
        return chunks

    def normalize(df): 
        min = df['high'].min()
        norm = (df['high'].max() - min)
        df['high'] = (df['high'] - min)/norm
        return df
    # Group by 'symbol' and apply the custom function
    grouped_df = df_sorted.groupby('symbol').apply(extract_consecutive_chunks)

    df_list = [item for sublist in grouped_df.values for item in sublist]
    df_list_norm = [normalize(df) for df in df_list]

    dataset = [torch.tensor(df.values).double() for df in df_list_norm]
    
    train_tensor, val_tensor = train_test_split(dataset, test_size=0.1, random_state=18)

    train_tensor = torch.stack(train_tensor)
    val_tensor = torch.stack(val_tensor)

    train_dataset = TensorDataset(train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)    
    
    val_dataset = TensorDataset(val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)    # each batch is [128,50,4]
    return train_dataloader, val_dataloader


def snp500C3(seq_length):
    df = pd.read_csv('Data\SP 500 Stock Prices 2014-2017.csv', header=None)
    df.columns = df.iloc[0]

    # Drop the first row
    df = df[1:].reset_index(drop=True)
    data = df.dropna()
    data['date'] = pd.to_datetime(data['date'])
    data['t0'] = (data['date'] - data['date'].min()).dt.days.astype(int)
    data = data.drop(['volume', 'date', 'open', 'close', 'low'], axis= 1)
    data['high'] = pd.to_numeric(data['high'])
    df_sorted = data.sort_values(by=['symbol', 't0'])

    # Define a function to extract consecutive chunks of 50 rows and drop 'symbol' and 't0'
    def extract_consecutive_chunks(group):
        chunks = [group.iloc[i:i+seq_length].reset_index(drop=True).drop(['symbol', 't0'], axis=1) for i in range(0, len(group), seq_length) if len(group[i:i+seq_length]) == seq_length]
        return chunks

    def normalize(df): 
        min = df['high'].min()
        norm = (df['high'].max() - min)
        df['high'] = (df['high'] - min)/norm
        return df
    
    # Group by 'symbol' and apply the custom function
    grouped_df = df_sorted.groupby('symbol').apply(extract_consecutive_chunks)

    df_list = [item for sublist in grouped_df.values for item in sublist]
    df_list_norm = [normalize(df) for df in df_list]

    def add_seq(df):
        df['next_high'] = df['high'].shift(-1)
        df = df.dropna()
        return df

    df_list_norm_wSeq = [add_seq(df) for df in df_list_norm]
    
    dataset = [torch.tensor(df.values).double() for df in df_list_norm_wSeq]
    
    train_tensor, val_tensor = train_test_split(dataset, test_size=0.1, random_state=18)

    train_tensor = torch.stack(train_tensor)
    val_tensor = torch.stack(val_tensor)

    train_dataset = TensorDataset(train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)    
    
    val_dataset = TensorDataset(val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)    # each batch is [128,50,4]
    return train_dataloader, val_dataloader

def snp500C4(seq_length):
    df = pd.read_csv('Data\SP 500 Stock Prices 2014-2017.csv', header=None)
    df.columns = df.iloc[0]

    # Drop the first row
    df = df[1:].reset_index(drop=True)
    data = df.dropna()
    data['date'] = pd.to_datetime(data['date'])
    data['t0'] = (data['date'] - data['date'].min()).dt.days.astype(int)
    data = data.drop(['volume', 'date', 'open', 'close', 'low'], axis= 1)
    data['high'] = pd.to_numeric(data['high'])
    df_sorted = data.sort_values(by=['symbol', 't0'])

    # Define a function to extract consecutive chunks of 50 rows and drop 'symbol' and 't0'
    def extract_consecutive_chunks(group):
        chunks = [group.iloc[i:i+seq_length].reset_index(drop=True).drop(['symbol', 't0'], axis=1) for i in range(0, len(group), seq_length) if len(group[i:i+seq_length]) == seq_length]
        return chunks

    def normalize(df): 
        min = df['high'].min()
        norm = (df['high'].max() - min)
        df['high'] = (df['high'] - min)/norm
        return df
    # Group by 'symbol' and apply the custom function
    grouped_df = df_sorted.groupby('symbol').apply(extract_consecutive_chunks)

    df_list = [item for sublist in grouped_df.values for item in sublist]
    df_list_norm = [normalize(df) for df in df_list]

    def add_seq(df):
        df['dup_high'] = df['high']
        df = df.dropna()
        return df

    df_list_norm_wSeq = [add_seq(df) for df in df_list_norm]
    
    dataset = [torch.tensor(df.values).double() for df in df_list_norm_wSeq]
    
    train_tensor, val_tensor = train_test_split(dataset, test_size=0.1, random_state=18)

    train_tensor = torch.stack(train_tensor)
    val_tensor = torch.stack(val_tensor)

    train_dataset = TensorDataset(train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)    
    
    val_dataset = TensorDataset(val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)    # each batch is [128,50,4]
    return train_dataloader, val_dataloader