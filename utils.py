import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
import random

def get_rand_stocks(num, df, seq_length):
    # Calculate the count of each unique value in 'group_column'
    value_counts = df['symbol'].value_counts()

# Filter the unique values based on the condition (at least 'k' instances)
    valid_values = value_counts.index[value_counts >= 1000].tolist()
    random_items = random.sample(valid_values, 3)
    ls = []
    for name in random_items:
        temp = df[df['symbol'] == name]
        max = value_counts.loc[name]
        throw = max % seq_length
        temp = temp.head(max - throw)
        ls.append(temp)
    return ls
    


    

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

def reconstruct(model, data, plot):
        if plot:
            torch.manual_seed(torch.initial_seed() % (2**32))
            for i in range(2):
                print(data.size(0))
                random_index = torch.randint(0, data.size(0), (1,)).item()
                # Select rows using the random indices
                temp = data[random_index, :].unsqueeze(0)
                output, _ = model.model(temp)
                # Convert tensors to numpy arrays for plotting

                original_data = temp.detach().numpy().flatten().tolist()
                reconstructed_data = output.detach().numpy().flatten().tolist()

                plt.plot(range(len(original_data)), original_data,label='Original Data')
                plt.plot(range(len(original_data)), reconstructed_data, label='Reconstructed Data')
                plt.title('Original vs Reconstructed Data')
                # Add labels and legend
                plt.xlabel('Sequence Indices')
                plt.ylabel('Values')
                plt.legend()

                # Show the plot
                plt.show()   
        else:
            ls_og = []
            ls_rec = []
            for i, lst in enumerate(data):
                temp = torch.tensor(lst[0].values).unsqueeze(0)
                output, _ = model.model(temp)
                og_signal = temp.detach().numpy().flatten()
                reconstructed_data = output.detach().numpy().flatten()
                og_signal = og_signal * lst[2]
                reconstructed_data = reconstructed_data * lst[2]
                og_signal = og_signal + lst[1]
                reconstructed_data = reconstructed_data + lst[1]
                og_signal = og_signal.tolist()
                reconstructed_data = reconstructed_data.tolist()
                ls_rec = np.concatenate((ls_rec, reconstructed_data), axis=0)
                ls_og = np.concatenate((ls_og, og_signal), axis=0)
            return [ls_og, ls_rec]





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
        fig, axes = plt.subplots(2, num_digits, figsize=(5 * num_digits, 5))   
        for i in range(len(dig_pair)):
            # Plot for the current digit pair
            axes[0][i].imshow(dig_pair[i][0], cmap='gray', interpolation='none', origin='lower')
            axes[0][i].set_title(f'Original Digit {i + 1}')
            axes[0][i].set_xlabel('X-axis')
            axes[0][i].set_ylabel('Y-axis')

            axes[1][i].imshow(dig_pair[i][1], cmap='gray', interpolation='none', origin='lower')
            axes[1][i].set_title(f'Reconstructed Digit {i + 1}')
            axes[1][i].set_xlabel('X-axis')
            axes[1][i].set_ylabel('Y-axis')

        plt.tight_layout()
        plt.show()