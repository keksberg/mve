#ifndef FSSR_IFN_WO_INFLUENCE_CUDA_HEADER
#define FSSR_IFN_WO_INFLUENCE_CUDA_HEADER

#include <cuda_runtime.h>

#include "fssr/iso_octree.h"

FSSR_NAMESPACE_BEGIN


// template <int N, typename T> struct Vec;
//
// template <int M, int N, typename T>
// using Matrix = Vec<M, Vec<N, T> >;
// // typedef Vec<M, Vec<N, T> > Matrix<M, N>;
//
// template <typename T>
// struct Vec<3, T>
// {
// 	T a0, a1, a2;
//
// 	template <typename S>
// 	__device__ __host__ __forceinline__
// 	Vec<3, T>& operator[](S idx) {
// 		switch (idx) {
// 			case 0:
// 				return a0;
// 			case 1:
// 				return a1;
// 			default:
// 				return a2;
// 		}
// 	}
// };


struct VoxelDataCUDA
{
    __device__ __host__ VoxelDataCUDA (void);

    float value;
    float conf;
    float scale;
    float3 color;
};

inline
__device__ __host__ VoxelDataCUDA::VoxelDataCUDA (void)
    : value(0.0f)
    , conf(0.0f)
    , scale(0.0f)
{
    color.x = 0.0f;
    color.y = 0.0f;
    color.z = 0.0f;
}

void compute_all_voxels_gpu_v1 (float3 *voxel_positions_d, uint num_voxels,
                                SampleCUDA *sample_vec_d, uint num_samples,
                                uint *sample_list_offset_d,
                                uint *sample_list_d, uint num_sample_ptrs,
                                VoxelDataCUDA *voxel_result_d);


FSSR_NAMESPACE_END

#endif /* FSSR_IFN_WO_INFLUENCE_CUDA_HEADER */
