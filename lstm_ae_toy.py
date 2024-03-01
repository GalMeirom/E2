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
# import gridsearchCV



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

# Define the LSTM Autoencoder model
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=input_size,num_layers=num_layers, batch_first=True)
       
    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)
        # Latent representation
        latent = hidden[-1]  # Take the last hidden state
        # Decoder
        decoded, _ = self.decoder(latent.unsqueeze(1).repeat(1, x.size(1), 1))
        return decoded, latent

def train_and_evaluate(input_size, hidden_size, num_layers ,learning_rate, clip_value, train_dataloader, val_dataloader, epochs):
    model = LSTMAutoencoder(input_size, hidden_size, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_dataloader:
            batch_data = batch[0].unsqueeze(2)
            output, _ = model(batch_data)
            loss = criterion(output, batch_data)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], last Training Loss: {loss.item():.4f}')
        #print(f'Epoch [{epoch+1}/{epochs}], AVG Training Loss: {avg_train_loss:.4f}')

        # Evaluate on the validation set
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch_data = batch[0].unsqueeze(2)
                output, _ = model(batch_data)
                val_loss = criterion(output, batch_data)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        model.train()
        #print(f'Epoch [{epoch+1}/{epochs}], AVG Validation Loss: {avg_val_loss:.4f}')

    return [model, avg_val_loss]

def reconstruct(model, dataloader):
    data = dataloader.dataset.tensors[0][:1].unsqueeze(2)
    output, _ = model(data)
    print("Original Data shape", data.shape)
    print("Reconstructed Data shape", output.shape)
    # Convert tensors to numpy arrays for plotting
    original_data = data.detach().numpy().flatten().tolist()
    reconstructed_data = output.detach().numpy().flatten().tolist()
    
    plt.plot(range(len(original_data)), original_data,label='Original Data')
    plt.plot(range(len(original_data)), reconstructed_data, label='Reconstructed Data')

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()

    # Show the plot
    plt.show()
    
    
def Q1C2():

    # Generate synthetic data for demonstration
    num_of_seq = 10000 
    seq_length = 50
    data = Data.syntheticData(num_of_seq, seq_length)  # 100 sequences, each of length 20
    # Split the data into 60/20/20
    train_tensor, temp_tensor = train_test_split(data, test_size=0.4, random_state=18)
    val_tensor, test_tensor = train_test_split(temp_tensor, test_size=0.5, random_state=18)
    
    # Create Train DataLoader
    train_dataset = TensorDataset(train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)    
    
    # Create Validation DataLoader
    val_dataset = TensorDataset(val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)    
    
    epochs = 100
    models_dict = {}
    # Grid search hyperparameters
    param_grid = {
    'input_size': [1],
    'hidden_size': [32, 40, 48],
    'learning_rate': [0.05, 0.0075, 0.01],
    'clip_value' : [0.5, 1.0, 1.5]
    }
    best_params = None

    for params in ParameterGrid(param_grid):
        models_dict[str(params)] = train_and_evaluate(**params, num_layers= 1, epochs= epochs ,train_dataloader=train_dataloader, val_dataloader= val_dataloader)
        print(f'{str(params)} - {models_dict[str(params)][1]}')
        print("Reconstructed Data for the best model")
        reconstruct(models_dict[str(params)][0], val_dataloader)
        
        # Recons
    best_params = min(models_dict, key=lambda k: models_dict[k][1])
    best_model = models_dict[best_params]

    print("Best Hyperparameters:", best_params)
    print("Best Validation Loss:", best_model[1])

Q1C2()