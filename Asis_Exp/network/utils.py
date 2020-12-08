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


def pairwise_distance(fm):
    B,C,N = fm.size()
    fm2 = fm.permute(0,2,1)  # [B,N,5]
    dist = []
    for i in range(B):
        f = fm2[i]
        l1 = torch.abs(torch.sub(f, f.unsqueeze(1))).sum(-1)  # [N,N]
        dist.append(l1)
    return torch.stack(dist)  # [B,N,N]

def knn_thre(adj_matrix, k=20, thre=0.5):
    B,N = adj_matrix.size(0), adj_matrix.size(1)
    neg_adj = - 1.0 * adj_matrix
    vals, nn_idx = torch.topk(neg_adj, k=k, dim=-1)

    to_add = torch.arange(start=0,end=N).view(-1,1)
    to_add = to_add.repeat(1,k)
    #print(to_add.size())

    final_nn_idx = []
    for i in range(B):
        idx_vals = vals[i]
        idx_nn_idx = nn_idx[i]
        mask = torch.tensor(idx_vals < -1.0 * thre, dtype=torch.int32)
        #print(mask.size())
        idx_to_add = to_add * mask
        idx_nn_idx = idx_nn_idx * (1 - mask) + idx_to_add
        final_nn_idx.append(idx_nn_idx)
    return torch.stack(final_nn_idx)  # [B,N,K]

def get_local_feature(fms_sem, nn_idx, k=20):
    B, C, N = fms_sem.size()
    fms_sem2 = fms_sem.permute(0,2,1)  # [B,N,C]

    idx_ = torch.arange(start=0, end=B) * N
    idx_ = idx_.view(B,1,1)

    fms_sem_flat = fms_sem2.contiguous().view(-1,C)
    final_idx = (nn_idx+idx_).long()
    return fms_sem_flat[final_idx]


if __name__ == '__main__':
    a = torch.randn(3,5,10)
    a = torch.tensor(a, requires_grad=True)
    dist = pairwise_distance(a)
    #print(dist)
    print(a.requires_grad, dist.requires_grad)