import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def shape(x):
    print(x.shape)

def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)

def get_max_indices(matrix):
    # Find the indices of the maximum values along each row
    max_indices = torch.argmax(matrix, dim=1, keepdim=True).type(torch.int)
    return max_indices

def reconstruct(model, data):
        random_index = torch.randint(0, data.size(0), (1,)).item()

        # Select rows using the random indices
        data = data[random_index, :].unsqueeze(0)
        output, _ = model.model(data)
        # Convert tensors to numpy arrays for plotting
        if model.input_size == 1:
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
        else:
            original_data = data.detach().numpy()#
            reconstructed_data = output.detach().numpy()#
            titles = ['Open', 'High', 'Low', 'Close']
            for i, name in enumerate(titles):
                original_feature = original_data[:,:,i].flatten().tolist()
                recon_feature = reconstructed_data[:,:,i].flatten().tolist()
                plt.plot(range(len(original_feature)), original_feature,label='Original ' + name)
                plt.plot(range(len(original_feature)), recon_feature, label='Reconstructed ' + name)
    
            # Add labels and legend
                plt.xlabel('Time Stamp')
                plt.ylabel('Normalized Values')
            plt.legend()
    
            # Show the plot
            plt.show() 


def reImage(model, dataloader):
    model.model.eval()
    nums = []
    it = iter(dataloader)
    temp = next(it)
    temp = temp[0].squeeze(1)
    random_indices = torch.randperm(temp.size(0))

    # Select the first three rows using the random indices
    temp = temp[random_indices[:3]]
    with torch.no_grad():
        num_digits = len(temp)
        dig_pair = []
        # Create a subplot with enough columns for each digit pair
        for i, dig in enumerate(temp):
            original_data = dig.numpy()
            reconstructed_data, _ = model.model(dig.unsqueeze(0))
            reconstructed_data = reconstructed_data.squeeze(0).numpy()
            dig_pair.append([original_data, reconstructed_data])
        fig, axes = plt.subplots(1, 2 * num_digits, figsize=(5 * num_digits, 5))   
        for i in range(len(dig_pair)):
            # Plot for the current digit pair
            axes[i * 2].imshow(dig_pair[i][0], cmap='gray', interpolation='none', origin='lower')
            axes[i * 2].set_title(f'Original Digit {i + 1}')
            axes[i * 2].set_xlabel('X-axis')
            axes[i * 2].set_ylabel('Y-axis')

            axes[i * 2 + 1].imshow(dig_pair[i][1], cmap='gray', interpolation='none', origin='lower')
            axes[i * 2 + 1].set_title(f'Reconstructed Digit {i + 1}')
            axes[i * 2 + 1].set_xlabel('X-axis')
            axes[i * 2 + 1].set_ylabel('Y-axis')

        plt.tight_layout()
        plt.show()