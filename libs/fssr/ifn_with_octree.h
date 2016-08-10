#ifndef FSSR_IFN_WITH_OCTREE_HEADER
#define FSSR_IFN_WITH_OCTREE_HEADER

#include <cuda_runtime.h>

#include "defines.h"
// #include "fssr/octree_cuda.h"
#include "cuda_types.h"

FSSR_NAMESPACE_BEGIN

void
ifn_with_octree_gpu(float3 root_center, float root_size,
                    CUTypes::Node *nodes, uint nodes_size,
                    CUTypes::Sample *samples, uint samples_size,
                    float3 *voxel_positions, uint voxel_positions_size,
                    CUTypes::VoxelData *result
                   );

__global__ void
ifn_with_octree_cuda(float3 root_center, float root_size,
                     const CUTypes::Node *__restrict__ nodes, uint nodes_size,
                     const CUTypes::Sample *__restrict__ samples, uint samples_size,
                     const float3 *__restrict__ voxel_positions, uint voxel_positions_size,
                     CUTypes::VoxelData *__restrict__ result,
                     float3 *__restrict__ timings
                    );


FSSR_NAMESPACE_END

#endif /* FSSR_IFN_WITH_OCTREE_HEADER */
