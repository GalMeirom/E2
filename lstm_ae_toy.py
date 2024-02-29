import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Define the LSTM Autoencoder model
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        # Latent representation
        self.latent = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.decoder = nn.LSTM(input_size=latent_size, hidden_size=hidden_size, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)

        # Latent representation
        latent = self.latent(hidden[-1])  # Take the last hidden state

        # Decoder
        decoded, _ = self.decoder(latent.unsqueeze(1).repeat(1, x.size(1), 1))

        # Output layer
        output = self.output_layer(decoded)

        return output, latent

# Step 1: Generate 10,000 input sequences of length 50 with values in the range [0, 1] as PyTorch tensors
num_samples = 10000
seq_length = 50
input_sequences = torch.rand(num_samples, seq_length)

# Step 2: Post-process each sequence
for sequence in input_sequences:
    i = torch.randint(20, 31, size=(1,))  # Sample i from [20, 30]
    start_index = max(0, i - 5)
    end_index = min(seq_length, i + 6)
    sequence[start_index:end_index] *= 0.1

# Step 3: Split the data
X_train_tensor, X_temp_tensor = train_test_split(input_sequences, test_size=0.4, random_state=18)
X_val_tensor, X_test_tensor = train_test_split(X_temp_tensor, test_size=0.5, random_state=18)


# # Step 4: Print some example sequences
# print("Example sequences from training set:")
# for i in range(3):
#     print("Sequence", i+1, ":", X_train_tensor[i])

# # Step 5: Plot some example sequences
# plt.figure(figsize=(10, 6))

# for i in range(3):
#     plt.subplot(3, 1, i+1)
#     plt.plot(X_train_tensor[i], label='Sequence {}'.format(i+1))
#     plt.xlabel('Time Step')
#     plt.ylabel('Value')
#     plt.title('Training Set - Sequence {}'.format(i+1))
#     plt.legend()

# plt.tight_layout()
# plt.show()

# Hyperparameters
input_size = 200
hidden_size = 32
latent_size = 4
learning_rate = 0.001
epochs = 1000

# Generate synthetic data for demonstration
data = torch.randn(100, 20, input_size)  # 100 sequences, each of length 20

# Create DataLoader
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Instantiate model and optimizer
model = LSTMAutoencoder(input_size, hidden_size, latent_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        batch_data = batch[0]

        # Forward pass
        output, latent = model(batch_data)

        # Compute loss
        loss = criterion(output, batch_data)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss for every epoch
    if epoch % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Example of using the trained model to get latent representation
sample_input = torch.randn(1, 20, input_size)  # Input sequence of length 20
_, latent_representation = model(sample_input)

print("Latent Representation:")
print(latent_representation.detach().numpy())
