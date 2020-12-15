import torch

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

    to_add = torch.arange(start=0,end=N).view(-1,1).cuda()
    to_add = to_add.repeat(1,k)

    final_nn_idx = []
    for i in range(B):
        idx_vals = vals[i]
        idx_nn_idx = nn_idx[i]
        # mask = (idx_vals < -1.0 * thre).clone.detach().requires_grad_(True)
        mask = torch.tensor(idx_vals < -1.0 * thre, dtype=torch.int32).cuda()
        idx_to_add = to_add * mask
        idx_nn_idx = idx_nn_idx * (1 - mask) + idx_to_add
        final_nn_idx.append(idx_nn_idx)
    return torch.stack(final_nn_idx)  # [B,N,K]

def get_local_feature(fms_sem, nn_idx, k=20):
    B, C, N = fms_sem.size()
    fms_sem2 = fms_sem.permute(0,2,1)  # [B,N,C]

    idx_ = torch.arange(start=0, end=B) * N
    idx_ = idx_.view(B,1,1).cuda()

    fms_sem_flat = fms_sem2.contiguous().view(-1,C)
    final_idx = (nn_idx+idx_).long()
    return fms_sem_flat[final_idx]

def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long().cuda()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).cuda().scatter_add_(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor