import torch
# Example tensor of shape (k, l, 1)
tensor = torch.randn((3, 4, 1))

# Drop the line at index 1 along the second dimension
tensor_after_drop = torch.cat((tensor[:, :0, :], tensor[:, 1:, :]), dim=1)

# Print the original and modified tensors
print("Original Tensor:")
print(tensor)

print("\nTensor after dropping line at index 1:")
print(tensor_after_drop)