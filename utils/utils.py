import torch
import numpy as np

def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))

# def gather_nd(params, indices):
#     """"
#     Gather slices from params into a Tensor with shape specified by indices.
#     example:
#     indices = [[0, 0], [1, 1]]
#     params = [['a', 'b'], ['c', 'd']]
#     output = ['a', 'd']
#     """
#     assert len(indices.shape) == len(params.shape)-1
#     assert indices.shape[-1] <= torch.matrix_rank(params)
#     return

def scatter_nd(indices, updates, shape):
    """the inverse of torch.gather function
    example:
    indices = [[0, 0], [1, 1]]
    updates = [4, 5]
    shape = (2,2)
    output = [[4, 0], [0, 5]]
    """
    output = torch.zeros(shape,dtype=torch.int64)
    for i, update in enumerate(updates):
        output[indices[i,0],indices[i,1]] = update
    return output

if __name__=='__main__':
    indices = torch.tensor([[0, 0], [1, 1]])
    updates = torch.tensor([4, 5])
    shape = (2, 2)
    output = scatter_nd(indices,updates, shape)
    print(output)
    assert output == torch.tensor([[4, 0], [0, 5]])


