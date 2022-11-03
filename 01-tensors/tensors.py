import torch

tensor_var = torch.tensor([1, 2, 3, 4, 5])
print(tensor_var)
print(type(tensor_var))

tensor_var2 = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)
print(tensor_var2)

sum_tensor = torch.sum(tensor_var2)
print(sum_tensor)

print(tensor_var2.shape)
squez_tensor = torch.squeeze(tensor_var2)
print(squez_tensor.shape)
unsquez_tensor = torch.unsqueeze(tensor_var2, dim=0)
print(unsquez_tensor.shape)