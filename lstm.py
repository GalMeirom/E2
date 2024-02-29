import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Define the LSTM Autoencoder model
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        # Encoder
        x, _ = self.encoder(x)
        
        # Decoder
        x, _ = self.decoder(x)

        return x