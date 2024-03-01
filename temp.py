import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.latent_size = latent_size

        # Decoder
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)
        latent_representation = hidden[-1]  # Use the last hidden state as the latent representation

        # Decoder
        latent_representation = latent_representation.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded_output, _ = self.decoder(latent_representation)

        return decoded_output, latent_representation

# Example usage
input_size = 1  # Input features
hidden_size = 5  # Hidden size of the LSTM layer
latent_size = 3  # Size of the latent representation
num_layers = 20   # Number of LSTM layers

# Create an instance of the LSTMAutoencoder
lstm_ae = LSTMAutoencoder(input_size, hidden_size, latent_size, num_layers)

# Define a sample input sequence (batch_size=1, sequence_length=10, input_features=10)
sample_input = torch.randn(1, 20, 1)

# Define a target sequence (in a real scenario, this would be your ground truth)
target = sample_input.clone()

# Hyperparameters
learning_rate = 0.001
num_epochs = 10000

# Optimizer and Loss function
optimizer = optim.Adam(lstm_ae.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Lists to store losses for plotting
losses = []

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    decoded_output, _ = lstm_ae(sample_input)

    # Compute the reconstruction loss
    loss = criterion(decoded_output, target)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Append the loss to the list for plotting
    losses.append(loss.item())

    # Print training information
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plotting the original and reconstructed data
plt.figure(figsize=(10, 5))

# Convert tensors to numpy arrays for plotting
original_data = target.squeeze().detach().numpy()
reconstructed_data = decoded_output.squeeze().detach().numpy()

plt.plot(original_data, label='Original Data', marker='o')
plt.plot(reconstructed_data, label='Reconstructed Data', marker='x')
plt.title('Original vs Reconstructed Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()