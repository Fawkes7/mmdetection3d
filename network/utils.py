import torch

def gather_nd(params, indices, name=None):
    '''
    the input indices must be a 2d tensor in the form of [[a,b,..,c],...],
    which represents the location of the elements.
    '''

    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    return torch.take(params, idx)


def scatter_nd(indices, value, shape):
    return None
