import pandas as pd
import numpy as np
import Data
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid
import utils as ut
import lstm
from sklearn.preprocessing import MinMaxScaler


# 0       symbol        date     open     high      low    close    volume
# 1          AAL  2014-01-02    25.07    25.82    25.06    25.36   8998943
# 2         AAPL  2014-01-02  79.3828  79.5756  78.8601  79.0185  58791957
# 3          AAP  2014-01-02   110.36   111.88   109.29   109.74    542711
# 4         ABBV  2014-01-02    52.12    52.33    51.52    51.98   4569061
# ...        ...         ...      ...      ...      ...      ...       ...
# 497468     XYL  2017-12-29    68.53     68.8    67.92     68.2   1046677
# 497469     YUM  2017-12-29    82.64    82.71    81.59    81.61   1347613
# 497470     ZBH  2017-12-29   121.75   121.95   120.62   120.67   1023624
# 497471    ZION  2017-12-29    51.28    51.55    50.81    50.83   1261916
# 497472     ZTS  2017-12-29    72.55    72.76    72.04    72.04   1704122

def C1():
    df = pd.read_csv('Data\SP 500 Stock Prices 2014-2017.csv', header=None)
    df.columns = df.iloc[0]

    # Drop the first row
    df = df[1:].reset_index(drop=True)
    data = df.dropna()
    googl_data = data[data['symbol'] == 'GOOGL']
    amzn_data = data[data['symbol'] == 'AMZN']
    googl_data = googl_data[['date', 'high']]
    amzn_data = amzn_data[['date', 'high']]
    goog_dates = googl_data['date'].values.astype(str)
    goog_high = googl_data['high'].values.astype(float)
    amzn_dates = amzn_data['date'].values.astype(str)
    amzn_high = amzn_data['high'].values.astype(float)

    goog_dates_numerical = np.arange(len(goog_dates))
    amzn_dates_numerical = np.arange(len(amzn_dates))

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))

    # Plot the first Y array on the first subplot
    ax1.plot(goog_dates_numerical, goog_high, label='Google High', color='blue')
    ax1.set_xticks(goog_dates_numerical[::100])
    ax1.set_xticklabels(goog_dates[::100], rotation=45, ha='right')  # Rotate labels for better visibility
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('Stock high')
    ax1.set_title('Google highs 2014 - 2017')
    ax1.legend()

    # Plot the second Y array on the second subplot
    ax2.plot(amzn_dates_numerical, amzn_high, label='Amazon High', color='red')
    ax2.set_xticks(amzn_dates_numerical[::100])
    ax2.set_xticklabels(amzn_dates[::100], rotation=45, ha='right')  # Rotate labels for better visibility
    ax2.set_xlabel('Dates')
    ax2.set_ylabel('Stock high')
    ax2.set_title('Amazon highs 2014 - 2017')
    ax2.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

def C2():
    seq_length = 40
    
    train_dataloader, val_dataloader = Data.snp500(seq_length)

    input_size = 1
    
    epochs = 300
    models_dict = {}

    param_grid = {
    'input_size': [input_size],
    'num_layers': [1],
    'hidden_size': [64],
    'learning_rate': [0.0002],
    'clip_value' : [1.0],
    'seq_length': [seq_length],
    'numC': [1],
    'apprx': [0],

    }

    for params in ParameterGrid(param_grid):
        print(str(params))
        model = lstm.LSTM_Model(**params)
        model.model.double()
        model.train(train_dataloader, epochs)
        models_dict[str(params)] = [model, model.eval(val_dataloader)]
        ut.reconstruct(models_dict[str(params)][0], val_dataloader.dataset.tensors[0])

def C3():
    seq_length = 20
    
    train_dataloader, val_dataloader = Data.snp500C3(seq_length+1)

    input_size = 1
    
    epochs = 10

    param_grid = {
    'input_size': [input_size],
    'num_layers': [1],
    'hidden_size': [64],
    'learning_rate': [0.0002],
    'clip_value' : [1.0],
    'seq_length': [seq_length],
    'numC': [1],
    'apprx': [1],

    }

    training_losses = []
    validation_losses = []
    for params in ParameterGrid(param_grid):
        model = lstm.LSTM_Model_wPred(**params)
        model.model.double()
        for i in range(epochs):
            avg_train_loss = model.train_approx(train_dataloader)
            print(f'Epoch [{i+1}/{epochs}], AVG Training Loss: {avg_train_loss}')
            training_losses.append(avg_train_loss)
            val_loss = model.val_approx(val_dataloader)
            print(f'Epoch [{i+1}/{epochs}], AVG Val Loss: {val_loss}')
            validation_losses.append(val_loss)

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))

        # Plot the first Y array on the first subplot
        ax1.plot(range(1, epochs+1), training_losses, label='Training Loss', color='blue')
        ax1.set_xlabel('Ephocs')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Average Training Loss vs ephocs')
        ax1.legend()

        # Plot the second Y array on the second subplot
        ax2.plot(range(1, epochs+1), validation_losses, label='Validation Loss', color='red')
        ax2.set_xlabel('Ephocs')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Average Validation Loss vs ephocs')
        ax2.legend()

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()

def C4():
    seq_length = 20
    
    train_dataloader, val_dataloader = Data.snp500C4(seq_length)

    input_size = 1
    
    epochs = 10

    param_grid = {
    'input_size': [input_size],
    'num_layers': [1],
    'hidden_size': [64],
    'learning_rate': [0.0002],
    'clip_value' : [1.0],
    'seq_length': [seq_length],
    'numC': [1],
    'apprx': [2],

    }

    training_losses = []
    validation_losses = []
    for params in ParameterGrid(param_grid):
        model = lstm.LSTM_Model_wPred(**params)
        model.model.double()
        for i in range(epochs):
            avg_train_loss = model.train_approx(train_dataloader)
            print(f'Epoch [{i+1}/{epochs}], AVG Training Loss: {avg_train_loss}')
            training_losses.append(avg_train_loss)
            val_loss = model.val_approx(val_dataloader)
            print(f'Epoch [{i+1}/{epochs}], AVG Val Loss: {val_loss}')
            validation_losses.append(val_loss)

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))

        # Plot the first Y array on the first subplot
        ax1.plot(range(1, epochs+1), training_losses, label='Training Loss', color='blue')
        ax1.set_xlabel('Ephocs')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Average Training Loss vs ephocs')
        ax1.legend()

        # Plot the second Y array on the second subplot
        ax2.plot(range(1, epochs+1), validation_losses, label='Validation Loss', color='red')
        ax2.set_xlabel('Ephocs')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Average Validation Loss vs ephocs')
        ax2.legend()

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()



C2()