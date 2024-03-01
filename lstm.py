import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)
        # Decoder
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=input_size,num_layers=num_layers, batch_first=True)
       
        self.linear = nn.Linear(seq_length, seq_length)

    def forward(self, x):
        # Encoder
        a, (hidden, _) = self.encoder(x)
        # Latent representation
        latent = hidden[-1]  # Take the last hidden state
        # Decoder
        #latent.unsqueeze(1).repeat(1,x.size(1),1)
        decoded, _ = self.decoder(latent.unsqueeze(1).repeat(1,x.size(1),1))
        
        rec_output = torch.tanh(self.linear(decoded.squeeze(2))).unsqueeze(2)

        return rec_output, _

class LSTM_Model():
    def __init__(self, input_size, hidden_size, num_layers, learning_rate, clip_value, seq_length):
        self.model = LSTMAutoencoder(input_size, hidden_size, num_layers, seq_length)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clip = clip_value
        self.input_size = input_size
    
    def train(self, train_dataloader, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0
            for batch in train_dataloader:
                batch_data = batch[0]
                output, _ = self.model(batch_data)
                loss = self.criterion(output, batch_data)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_dataloader)
            if epoch % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], AVG Training Loss: {avg_train_loss}')

    def eval(self, val_dataloader):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch_data = batch[0]
                output, (h, c) = self.model(batch_data)
                val_loss = self.criterion(output, batch_data)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f'AVG VAL Loss: {avg_val_loss}')
        return avg_val_loss