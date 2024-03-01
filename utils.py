import torch
import matplotlib.pyplot as plt


def shape(x):
    print(x.shape)

def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)

def reconstruct(model, data):
        random_index = torch.randint(0, data.size(0), (1,)).item()

        # Select rows using the random indices
        data = data[random_index, :].unsqueeze(0)
        output, _ = model.model(data)
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