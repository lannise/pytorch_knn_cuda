#include <THC/THC.h>
#include "knn_cuda_kernel.h"

extern THCState *state;

int knn(THCudaTensor *ref_tensor, THCudaTensor *query_tensor,
    THCudaLongTensor *idx_tensor, THCudaTensor *dist_tensor) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 4, idx_tensor, dist_tensor, ref_tensor, query_tensor));
  long batch, ref_nb, query_nb, dim, k;
  THArgCheck(THCudaTensor_nDimension(state, ref_tensor) == 3 , 0, "ref_tensor: 3D Tensor expected");
  THArgCheck(THCudaTensor_nDimension(state, query_tensor) == 3 , 1, "query_tensor: 3D Tensor expected");
  THArgCheck(THCudaLongTensor_nDimension(state, idx_tensor) == 3 , 3, "idx_tensor: 3D Tensor expected");
  THArgCheck(THCudaTensor_nDimension(state, dist_tensor) == 3 , 4, "dist_tensor: 3D Tensor expected");
  THArgCheck(THCudaTensor_size(state, ref_tensor, 0) == THCudaTensor_size(state, query_tensor,0), 0, "input sizes must match");
  THArgCheck(THCudaTensor_size(state, ref_tensor, 1) == THCudaTensor_size(state, query_tensor,1), 0, "input sizes must match");
  THArgCheck(THCudaLongTensor_size(state, idx_tensor, 1) == THCudaTensor_size(state, dist_tensor,1), 0, "output sizes must match");

  ref_tensor = THCudaTensor_newContiguous(state, ref_tensor);
  query_tensor = THCudaTensor_newContiguous(state, query_tensor);

  batch = THCudaLongTensor_size(state, ref_tensor, 0);
  dim = THCudaTensor_size(state, ref_tensor, 1);
  k = THCudaLongTensor_size(state, idx_tensor, 1);
  ref_nb = THCudaTensor_size(state, ref_tensor, 2);
  query_nb = THCudaTensor_size(state, query_tensor, 2);

  float *ref_dev = THCudaTensor_data(state, ref_tensor);
  float *query_dev = THCudaTensor_data(state, query_tensor);
  long *idx_dev = THCudaLongTensor_data(state, idx_tensor);
  float *dist_dev = THCudaTensor_data(state, dist_tensor);
  
  for (int b = 0; b < batch; b++) {
    knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k, 
      dist_dev + b * k * query_nb, idx_dev + b * k * query_nb, THCState_getCurrentStream(state));
  }
  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in knn: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  return 1;
}
