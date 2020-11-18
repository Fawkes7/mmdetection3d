import torch

indices = torch.tensor([[0, 1], [2, 3]])
updates = torch.tensor([[5, 5, 5, 5],
                        [6, 6, 6, 6]])
result = torch.zeros((4, 4, 4), dtype = torch.int64)
result[indices[:, 0], indices[:, 1]] = updates
print("PyTorch")
print(result.numpy())
