import torch

def get_max_indices(matrix):
    # Find the indices of the maximum values along each row
    max_indices = torch.argmax(matrix, dim=1, keepdim=True)
    
    return max_indices

# Example usage:
# Create a sample matrix
sample_matrix = torch.tensor([[1, 3, 5],
                             [2, 8, 6],
                             [0, 7, 4]])

# Get the matrix of maximum indices
result_matrix = get_max_indices(sample_matrix)

print("Original Matrix:")
print(sample_matrix)
print("\nMatrix of Maximum Indices:")
print(result_matrix)
print(result_matrix.shape)