#include <stdio.h>
#include <stdlib.h>
#include "torch_utils.h"
#include "cuda_utils.h"

/*
  Function: get how many points in each voxel grid
  Args:
    b      : batch size
    n      : number of points
    r      : voxel resolution
    r2     : = r * r
    r3     : s, voxel cube size = r ** 3
    coords : coords of each point, IntTensor[b, 3, n]
    ind    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
*/

__global__ void grid_stats_kernel(int b, int n, int r, int r2, int r3,
                                  const int *__restrict__ coords,
                                  int *__restrict__ ind, int *cnt) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  ind += batch_index * n;
  cnt += batch_index * r3;

  for (int i = index; i < n; i += stride) {
    // if (ind[i] == -1)
    //   continue;
    ind[i] = coords[i] * r2 + coords[i + n] * r + coords[i + n + n];
    atomicAdd(cnt + ind[i], 1);
  }
}

/*
  Function: average pool voxelization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    ind : voxel index of each point, IntTensor[b, n]
    cnt : #points in each voxel index, IntTensor[b, s]
    feat: features, FloatTensor[b, c, n]
    out : outputs, FloatTensor[b, c, s]
*/
__global__ void avg_voxelize_kernel(int b, int c, int n, int s,
                                    const int *__restrict__ ind,
                                    const int *__restrict__ cnt,
                                    const float *__restrict__ feat,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  feat += batch_index * c * n;
  out += batch_index * c * s;
  cnt += batch_index * s;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    // if (pos == -1)
    //   continue;
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(out + j * s + pos, feat[j * n + i] * div_cur_cnt);
      }
    }
  }
}

/*
  Function: average pool voxelization (backward)
  Args:
    b      : batch size
    c      : #channels
    n      : number of points
    r3     : voxel cube size = voxel resolution ** 3
    ind    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
    grad_y : grad outputs, FloatTensor[b, c, s]
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
__global__ void avg_voxelize_grad_kernel(int b, int c, int n, int r3,
                                         const int *__restrict__ ind,
                                         const int *__restrict__ cnt,
                                         const float *__restrict__ grad_y,
                                         float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  grad_x += batch_index * c * n;
  grad_y += batch_index * c * r3;
  cnt += batch_index * r3;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    // if (pos == -1)
    //   continue;
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(grad_x + j * n + i, grad_y[j * r3 + pos] * div_cur_cnt);
      }
    }
  }
}

void avg_voxelize(int b, int c, int n, int r, int r2, int r3, const int *coords,
                  const float *feat, int *ind, int *cnt, float *out) {
  grid_stats_kernel<<<b, optimal_num_threads(n)>>>(b, n, r, r2, r3, coords, ind,
                                                   cnt);
  avg_voxelize_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, r3, ind, cnt,
                                                     feat, out);
  CUDA_CHECK_ERRORS();
}

void avg_voxelize_grad(int b, int c, int n, int s, const int *ind,
                       const int *cnt, const float *grad_y, float *grad_x) {
  avg_voxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, s, ind, cnt,
                                                          grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}

namespace voxelization {
std::vector<at::Tensor> avg_voxelize_forward_gpu(const at::Tensor features,
                                             const at::Tensor coords,
                                             const int resolution) {
  CHECK_CUDA(features);
  CHECK_CUDA(coords);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(coords);
  CHECK_IS_FLOAT(features);
  CHECK_IS_INT(coords);

  int b = features.size(0);
  int c = features.size(1);
  int n = features.size(2);
  int r = resolution;
  int r2 = r * r;
  int r3 = r2 * r;
  at::Tensor ind = torch::zeros(
      {b, n}, at::device(features.device()).dtype(at::ScalarType::Int));
  at::Tensor out = torch::zeros(
      {b, c, r3}, at::device(features.device()).dtype(at::ScalarType::Float));
  at::Tensor cnt = torch::zeros(
      {b, r3}, at::device(features.device()).dtype(at::ScalarType::Int));
  avg_voxelize(b, c, n, r, r2, r3, coords.data_ptr<int>(),
               features.data_ptr<float>(), ind.data_ptr<int>(),
               cnt.data_ptr<int>(), out.data_ptr<float>());
  return {out, ind, cnt};
}

/*
  Function: average pool voxelization (backward)
  Args:
    grad_y : grad outputs, FloatTensor[b, c, s]
    indices: voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
  Return:
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
at::Tensor avg_voxelize_backward_gpu(const at::Tensor grad_y,
                                 const at::Tensor indices,
                                 const at::Tensor cnt) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(indices);
  CHECK_CUDA(cnt);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(cnt);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_INT(indices);
  CHECK_IS_INT(cnt);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int s = grad_y.size(2);
  int n = indices.size(1);
  at::Tensor grad_x = torch::zeros(
      {b, c, n}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  avg_voxelize_grad(b, c, n, s, indices.data_ptr<int>(), cnt.data_ptr<int>(),
                    grad_y.data_ptr<float>(), grad_x.data_ptr<float>());
  return grad_x;
}

}