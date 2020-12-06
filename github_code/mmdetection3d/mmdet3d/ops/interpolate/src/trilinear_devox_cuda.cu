#include <stdio.h>
#include <stdlib.h>

#include "torch_utils.h"
#include "cuda_utils.h"

/*
  Function: trilinear devoxlization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    r   : voxel resolution
    r2  : r ** 2
    r3  : r ** 3
    coords : the coordinates of points, FloatTensor[b, 3, n]
    feat   : features, FloatTensor[b, c, r3]
    inds   : the voxel indices of point cube, IntTensor[b, 8, n]
    wgts   : weight for trilinear interpolation, FloatTensor[b, 8, n]
    outs   : outputs, FloatTensor[b, c, n]
*/
__global__ void trilinear_devoxelize_kernel(int b, int c, int n, int r, int r2,
                                            int r3, bool is_training,
                                            const float *__restrict__ coords,
                                            const float *__restrict__ feat,
                                            int *__restrict__ inds,
                                            float *__restrict__ wgts,
                                            float *__restrict__ outs) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  inds += batch_index * n * 8;
  wgts += batch_index * n * 8;
  feat += batch_index * c * r3;
  outs += batch_index * c * n;

  for (int i = index; i < n; i += stride) {
    float x = coords[i];
    float y = coords[i + n];
    float z = coords[i + n + n];
    float x_lo_f = floorf(x);
    float y_lo_f = floorf(y);
    float z_lo_f = floorf(z);

    float x_d_1 = x - x_lo_f; // / (x_hi_f - x_lo_f + 1e-8f)
    float y_d_1 = y - y_lo_f;
    float z_d_1 = z - z_lo_f;
    float x_d_0 = 1.0f - x_d_1;
    float y_d_0 = 1.0f - y_d_1;
    float z_d_0 = 1.0f - z_d_1;

    float wgt000 = x_d_0 * y_d_0 * z_d_0;
    float wgt001 = x_d_0 * y_d_0 * z_d_1;
    float wgt010 = x_d_0 * y_d_1 * z_d_0;
    float wgt011 = x_d_0 * y_d_1 * z_d_1;
    float wgt100 = x_d_1 * y_d_0 * z_d_0;
    float wgt101 = x_d_1 * y_d_0 * z_d_1;
    float wgt110 = x_d_1 * y_d_1 * z_d_0;
    float wgt111 = x_d_1 * y_d_1 * z_d_1;

    int x_lo = static_cast<int>(x_lo_f);
    int y_lo = static_cast<int>(y_lo_f);
    int z_lo = static_cast<int>(z_lo_f);
    int x_hi = (x_d_1 > 0) ? -1 : 0;
    int y_hi = (y_d_1 > 0) ? -1 : 0;
    int z_hi = (z_d_1 > 0) ? 1 : 0;

    int idx000 = x_lo * r2 + y_lo * r + z_lo;
    int idx001 = idx000 + z_hi;      // x_lo * r2 + y_lo * r + z_hi;
    int idx010 = idx000 + (y_hi & r);  // x_lo * r2 + y_hi * r + z_lo;
    int idx011 = idx010 + z_hi;      // x_lo * r2 + y_hi * r + z_hi;
    int idx100 = idx000 + (x_hi & r2); // x_hi * r2 + y_lo * r + z_lo;
    int idx101 = idx100 + z_hi;      // x_hi * r2 + y_lo * r + z_hi;
    int idx110 = idx100 + (y_hi & r);  // x_hi * r2 + y_hi * r + z_lo;
    int idx111 = idx110 + z_hi;      // x_hi * r2 + y_hi * r + z_hi;

    if (is_training) {
      wgts[i] = wgt000;
      wgts[i + n] = wgt001;
      wgts[i + n * 2] = wgt010;
      wgts[i + n * 3] = wgt011;
      wgts[i + n * 4] = wgt100;
      wgts[i + n * 5] = wgt101;
      wgts[i + n * 6] = wgt110;
      wgts[i + n * 7] = wgt111;
      inds[i] = idx000;
      inds[i + n] = idx001;
      inds[i + n * 2] = idx010;
      inds[i + n * 3] = idx011;
      inds[i + n * 4] = idx100;
      inds[i + n * 5] = idx101;
      inds[i + n * 6] = idx110;
      inds[i + n * 7] = idx111;
    }

    for (int j = 0; j < c; j++) {
      int jr3 = j * r3;
      outs[j * n + i] =
          wgt000 * feat[jr3 + idx000] + wgt001 * feat[jr3 + idx001] +
          wgt010 * feat[jr3 + idx010] + wgt011 * feat[jr3 + idx011] +
          wgt100 * feat[jr3 + idx100] + wgt101 * feat[jr3 + idx101] +
          wgt110 * feat[jr3 + idx110] + wgt111 * feat[jr3 + idx111];
    }
  }
}

/*
  Function: trilinear devoxlization (backward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    r3  : voxel cube size = voxel resolution ** 3
    inds   : the voxel indices of point cube, IntTensor[b, 8, n]
    wgts   : weight for trilinear interpolation, FloatTensor[b, 8, n]
    grad_y : grad outputs, FloatTensor[b, c, n]
    grad_x : grad inputs, FloatTensor[b, c, r3]
*/
__global__ void trilinear_devoxelize_grad_kernel(
    int b, int c, int n, int r3, const int *__restrict__ inds,
    const float *__restrict__ wgts, const float *__restrict__ grad_y,
    float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  inds += batch_index * n * 8;
  wgts += batch_index * n * 8;
  grad_x += batch_index * c * r3;
  grad_y += batch_index * c * n;

  for (int i = index; i < n; i += stride) {
    int idx000 = inds[i];
    int idx001 = inds[i + n];
    int idx010 = inds[i + n * 2];
    int idx011 = inds[i + n * 3];
    int idx100 = inds[i + n * 4];
    int idx101 = inds[i + n * 5];
    int idx110 = inds[i + n * 6];
    int idx111 = inds[i + n * 7];
    float wgt000 = wgts[i];
    float wgt001 = wgts[i + n];
    float wgt010 = wgts[i + n * 2];
    float wgt011 = wgts[i + n * 3];
    float wgt100 = wgts[i + n * 4];
    float wgt101 = wgts[i + n * 5];
    float wgt110 = wgts[i + n * 6];
    float wgt111 = wgts[i + n * 7];

    for (int j = 0; j < c; j++) {
      int jr3 = j * r3;
      float g = grad_y[j * n + i];
      atomicAdd(grad_x + jr3 + idx000, wgt000 * g);
      atomicAdd(grad_x + jr3 + idx001, wgt001 * g);
      atomicAdd(grad_x + jr3 + idx010, wgt010 * g);
      atomicAdd(grad_x + jr3 + idx011, wgt011 * g);
      atomicAdd(grad_x + jr3 + idx100, wgt100 * g);
      atomicAdd(grad_x + jr3 + idx101, wgt101 * g);
      atomicAdd(grad_x + jr3 + idx110, wgt110 * g);
      atomicAdd(grad_x + jr3 + idx111, wgt111 * g);
    }
  }
}

void trilinear_devoxelize(int b, int c, int n, int r, int r2, int r3,
                          bool training, const float *coords, const float *feat,
                          int *inds, float *wgts, float *outs) {
  trilinear_devoxelize_kernel<<<b, optimal_num_threads(n)>>>(
      b, c, n, r, r2, r3, training, coords, feat, inds, wgts, outs);
  CUDA_CHECK_ERRORS();
}

void trilinear_devoxelize_grad(int b, int c, int n, int r3, const int *inds,
                               const float *wgts, const float *grad_y,
                               float *grad_x) {
  trilinear_devoxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(
      b, c, n, r3, inds, wgts, grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}

std::vector<at::Tensor>
trilinear_devoxelize_forward(const int r, const bool is_training,
                             const at::Tensor coords,
                             const at::Tensor features) {
  CHECK_CUDA(features);
  CHECK_CUDA(coords);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(coords);
  CHECK_IS_FLOAT(features);
  CHECK_IS_FLOAT(coords);

  int b = features.size(0);
  int c = features.size(1);
  int n = coords.size(2);
  int r2 = r * r;
  int r3 = r2 * r;
  at::Tensor outs = torch::zeros(
      {b, c, n}, at::device(features.device()).dtype(at::ScalarType::Float));
  if (is_training) {
    at::Tensor inds = torch::zeros(
        {b, 8, n}, at::device(features.device()).dtype(at::ScalarType::Int));
    at::Tensor wgts = torch::zeros(
        {b, 8, n}, at::device(features.device()).dtype(at::ScalarType::Float));
    trilinear_devoxelize(b, c, n, r, r2, r3, true, coords.data_ptr<float>(),
                         features.data_ptr<float>(), inds.data_ptr<int>(),
                         wgts.data_ptr<float>(), outs.data_ptr<float>());
    return {outs, inds, wgts};
  } else {
    at::Tensor inds = torch::zeros(
        {1}, at::device(features.device()).dtype(at::ScalarType::Int));
    at::Tensor wgts = torch::zeros(
        {1}, at::device(features.device()).dtype(at::ScalarType::Float));
    trilinear_devoxelize(b, c, n, r, r2, r3, false, coords.data_ptr<float>(),
                         features.data_ptr<float>(), inds.data_ptr<int>(),
                         wgts.data_ptr<float>(), outs.data_ptr<float>());
    return {outs, inds, wgts};
  }
}

/*
  Function: trilinear devoxelization (backward)
  Args:
    grad_y  : grad outputs, FloatTensor[b, c, n]
    indices : the voxel coordinates of point cube, IntTensor[b, 8, n]
    weights : weight for trilinear interpolation, FloatTensor[b, 8, n]
    r       : voxel resolution
  Return:
    grad_x     : grad inputs, FloatTensor[b, c, s], s = r ** 3
*/
at::Tensor trilinear_devoxelize_backward(const at::Tensor grad_y,
                                         const at::Tensor indices,
                                         const at::Tensor weights,
                                         const int r) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(weights);
  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(weights);
  CHECK_CONTIGUOUS(indices);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_FLOAT(weights);
  CHECK_IS_INT(indices);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int n = grad_y.size(2);
  int r3 = r * r * r;
  at::Tensor grad_x = torch::zeros(
      {b, c, r3}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  trilinear_devoxelize_grad(b, c, n, r3, indices.data_ptr<int>(),
                            weights.data_ptr<float>(), grad_y.data_ptr<float>(),
                            grad_x.data_ptr<float>());
  return grad_x;
}