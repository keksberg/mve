#include "ifn_with_octree.h"
#include "basis_function_cuda.cuh"
#include <stdio.h>

#include <algorithm>
#include <iostream>

// GPU configuration
#define GRID_MAX 65535
#define SAMPLE_CACHE_SIZE 512
#define BLOCK_SIZE 128

// constants
#define MATH_SQRT3_CU      1.7320508075688772935274463415058723669f // sqrt(3)
#define INFLUENCE_FACTOR 3.0f

// switches
//#define SYNC
//#define SCALE_SELECTION
//#define TIMINGS

/*
 * TODO: Change octree design from:
 *   4--------5
 *  /|       /|
 * 0--------1 |
 * | |      | |
 * | |      | |
 * | 6------|-7
 * |/       |/
 * 2--------3
 *
 * to:
 *   5--------4
 *  /|       /|
 * 0--------1 |
 * | |      | |
 * | |      | |
 * | 6------|-3
 * |/       |/
 * 7--------2
 *
 * and do some magic :)
 *
 * [Max' 3D code drawings (TM)]
 */

FSSR_NAMESPACE_BEGIN

__host__ __device__ __forceinline__ uint
div_up(uint a, uint b)
{
    return (a + b - 1) / b;
}

__global__ void
ifn_with_octree_1VPT_cuda(float3 root_center, float root_size,
                          const CUTypes::Node *__restrict__ nodes, uint nodes_size,
                          const CUTypes::Sample *__restrict__ samples, uint samples_size,
                          const float3 *__restrict__ voxel_positions, uint voxels_size,
                          CUTypes::VoxelData *__restrict__ result,
                          float3 *__restrict__ timings
)
{

    uint block_id        = blockIdx.x  + blockIdx.y  * gridDim.x;
    uint thread_id       = threadIdx.x + threadIdx.y * blockDim.x;
    uint total_block_dim = blockDim.x  * blockDim.y;
    uint voxel_idx       = block_id * total_block_dim + thread_id;

    uint sample_cache[SAMPLE_CACHE_SIZE];
//    __shared__ int shared_current_octant[20 * BLOCK_SIZE];
//    int *current_octant = shared_current_octant + 20 * thread_id;
    int current_octant[20];

    if (voxel_idx >= voxels_size)
        return;

    float3 pos = voxel_positions[voxel_idx];


    float total_value        = 0.0f;
    float total_conf         = 0.0f;
    float total_scale        = 0.0f;
    float total_color_r      = 0.0f;
    float total_color_g      = 0.0f;
    float total_color_b      = 0.0f;
    float total_color_weight = 0.0f;

    uint influencing_samples_count = 0;
#ifdef TIMINGS
    uint time_a = 0, time_b = 0, time_c = 0;
#endif

    /* Influence query */
    uint i = 0;
    uint j = 0;
    uint count_processed;
    current_octant[0] = 0; // clear initial level
    int level = 0;

    float3 geom_center = root_center;
    float geom_size = root_size;

    do {
#ifdef TIMINGS
        clock_t timing_0 = clock();
#endif
        // NOTE: Begin of influence_query
        count_processed = 0;

        while (i < nodes_size)
        {
            CUTypes::Node node = nodes[i]; // NOTE: Load from global memory

            /* Estimate for the minimum distance. No sample is closer to pos. */
            float3 min_dist_vec = make_float3(pos.x - geom_center.x,
                                              pos.y - geom_center.y,
                                              pos.z - geom_center.z);
            float min_dist = sqrtf(min_dist_vec.x * min_dist_vec.x +
                                   min_dist_vec.y * min_dist_vec.y +
                                   min_dist_vec.z * min_dist_vec.z) - MATH_SQRT3_CU * geom_size / 2.0f;

            float max_scale = geom_size * 2.0f;
            bool jump = min_dist > max_scale * INFLUENCE_FACTOR;

            if (!jump)
            {
                // sample stuff
                /* Node could not be ruled out. Test all samples. */
                while (j < node.sample_length)
                {
                    uint sample_idx = node.sample_offset + j++;
                    const CUTypes::Sample sample = samples[sample_idx]; // NOTE: Load from global memory
                    float3 dist = make_float3(pos.x - sample.pos.x, pos.y - sample.pos.y, pos.z - sample.pos.z);

                    if (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z > INFLUENCE_FACTOR * INFLUENCE_FACTOR * sample.scale * sample.scale)
                        continue;

                    // NOTE: Write only #SAMPLE_CACHE_SIZE samples to the cache
                    sample_cache[count_processed++] = sample_idx; // NOTE: Write to shared memory

                    if (count_processed == SAMPLE_CACHE_SIZE) {
                        // SAMPLE_CACHE_SIZE reached, return to reduction
                        goto SAMPLE_IFN;
                    }
                }
                j = 0; // reset index j
            }
            // jump logic
            i += jump ? node.node_count : 1;

            // NOTE: At this point i is already the index of the next node
#ifdef SYNC
            __syncthreads();
#endif
            int octant = current_octant[level];
            float offset;
            if (node.node_count == 1 || jump)
            {
                // leaf, don't descend! go to the next octant

                // reset center
                /*
                 * BEFORE: (X = geom_center)  AFTER:
                 * +--+--+--+--+              +--+--+--+--+
                 * |  |  |  |  |              |  |  |  |  |
                 * +--X--+--+--+              +--+--+--+--+
                 * |  |  |  |  |              |  |  |  |  |
                 * +--+--+--+--+              +--+--X--+--+
                 * |     |     |              |     |     |
                 * |     |     |              |     |     |
                 * |     |     |              |     |     |
                 * +-----------+              +-----------+
                 * |gsize|  (gsize = geom_size)                        <-- scale
                 * |o.|     (o.    = offset    = geom_size / 2)        <-- scale
                 */
                offset = geom_size / 2.0f;
                while (true)
                {
                    // reset center to the middle of the parent
                    geom_center.x -= ((octant & 1) ? offset : -offset);
                    geom_center.y -= ((octant & 2) ? offset : -offset);
                    geom_center.z -= ((octant & 4) ? offset : -offset);

                    octant = ++current_octant[level];
                    if (octant != 8)
                        break;

                    // reached the last octant, we have to ascend
                    --level; // ascend
                    octant = current_octant[level]; // update octant register
                    geom_size *= 2.0f;
                    offset *= 2.0f; // update offset to new geom_size
                }

                // apply new offset
                /*
                * BEFORE: (X = geom_center)  AFTER:
                * +--+--+--+--+              +--+--+--+--+
                * |  |  |  |  |              |  |  |  |  |
                * +--+--+--+--+              +--+--+--X--+
                * |  |  |  |  |              |  |  |  |  |
                * +--+--X--+--+              +--+--+--+--+
                * |     |     |              |     |     |
                * |     |     |              |     |     |
                * |     |     |              |     |     |
                * +-----------+              +-----------+
                */
            }
            else {
                // descend into octree
                /*
                * BEFORE: (X = geom_center)  AFTER:
                * +--+--+--+--+              +--+--+--+--+
                * |  |  |  |  |              |  |  |  |  |
                * +--+--+--+--+              +--X--+--+--+
                * |  |  |  |  |              |  |  |  |  |
                * +--+--X--+--+              +--+--+--+--+
                * |     |     |              |     |     |
                * |     |     |              |     |     |
                * |     |     |              |     |     |
                * +-----------+              +-----------+
                * |gsize      |  (gsize = geom_size)                        <-- scale
                * |o.|           (o.    = offset    = geom_size / 4)        <-- scale
                *
                * [Max' fancy code drawings (TM)]
                */

                ++level; // descend
                octant = current_octant[level] = 0; // clear next level
                geom_size /= 2.0f;

                // descend geom
                offset = geom_size / 2.0;
            }
#ifdef SYNC
            __syncthreads();
#endif
            geom_center.x += ((octant & 1) ? offset : -offset);
            geom_center.y += ((octant & 2) ? offset : -offset);
            geom_center.z += ((octant & 4) ? offset : -offset);
        }

SAMPLE_IFN:
        // NOTE: End of influence_query

        __syncthreads();

        if (count_processed == 0)
            break;

        influencing_samples_count += count_processed;
#ifdef TIMINGS
        clock_t timing_1 = clock();
#endif

#ifdef SCALE_SELECTION
        // partial selection sort
        uint min_index;
        float min_value;
        uint percentile = count_processed / 100;
        for (uint i = 0; i < percentile; ++i)
        {
            min_index = i;
            min_value = samples[sample_cache[i]].scale;
            for (uint j = i + 1; j < count_processed; ++j)
            {
                float scale_j = samples[sample_cache[j]].scale;
                if (scale_j < min_value)
                {
                    min_index = j;
                    min_value = scale_j;
                }
                uint tmp = sample_cache[i];
                sample_cache[i] = sample_cache[min_index];
                sample_cache[min_index] = tmp;
            }
        }

        // threshold
        float max_scale = samples[sample_cache[percentile]].scale * 2.0f;

        __syncthreads();
#endif


#ifdef TIMINGS
        clock_t timing_2 = clock();
#endif

        // writing back to (the same) cache :) we don't need the old values anymore -> this saves lots of shared memory
        for (uint k = 0; k < count_processed; ++k)
        {
            const CUTypes::Sample sample = samples[sample_cache[k]]; // NOTE: Load from local memory

#ifdef SCALE_SELECTION
            if (sample.scale > max_scale)
                continue;
#endif

            // sample ifn
            float3 tpos = CUBasisFunction::transform_position(pos, sample.pos, sample.normal);

            /* Evaluate basis fucntion. */
            float value = CUBasisFunction::gaussian_derivative(sample.scale, tpos);

            /* Evaluate weight function. */
            float weight = CUBasisFunction::weighting_function(sample.scale, tpos) * sample.confidence;
            float color_weight = CUBasisFunction::gaussian_normalized(sample.scale / 5.0f, tpos) * sample.confidence;

            /* Incrementally update. */
            total_value        += value           * weight;
            total_conf         += weight;
            total_scale        += sample.scale    * color_weight;
            total_color_r      += sample.color.x * color_weight;
            total_color_g      += sample.color.y * color_weight;
            total_color_b      += sample.color.z * color_weight;
            total_color_weight += color_weight;
        }

#ifdef SYNC
        __syncthreads(); // maybe remove (?)
#endif
#ifdef TIMINGS
        clock_t timing_3 = clock();
        time_a += (timing_1 - timing_0) / 1000;
        time_b += (timing_2 - timing_1) / 1000;
        time_c += (timing_3 - timing_2) / 1000;
#endif
    }
    while (count_processed == SAMPLE_CACHE_SIZE);

#ifdef TIMINGS
    float sum = time_a + time_b + time_c;
    if (sum != 0.0f)
    {
        atomicAdd(&(timings->x), time_a / sum / voxels_size);
        atomicAdd(&(timings->y), time_b / sum / voxels_size);
        atomicAdd(&(timings->z), time_c / sum / voxels_size);
    }
#endif

    result[voxel_idx].value   = (influencing_samples_count == 0) ? 0.0f : total_value   / total_conf;
    result[voxel_idx].conf    = (influencing_samples_count == 0) ? 0.0f : total_conf;
    result[voxel_idx].scale   = (influencing_samples_count == 0) ? 0.0f : total_scale   / total_color_weight;
    result[voxel_idx].color.x = (influencing_samples_count == 0) ? 0.0f : total_color_r / total_color_weight;
    result[voxel_idx].color.y = (influencing_samples_count == 0) ? 0.0f : total_color_g / total_color_weight;
    result[voxel_idx].color.z = (influencing_samples_count == 0) ? 0.0f : total_color_b / total_color_weight;
}

/* This functions calls the CUDA kernel */
void
ifn_with_octree_gpu(float3 root_center, float root_size,
                    CUTypes::Node *nodes, uint nodes_size,
                    CUTypes::Sample *samples, uint samples_size,
                    float3 *voxel_positions, uint voxels_size,
                    CUTypes::VoxelData *result)
{
//    dim3 block(SAMPLE_CACHE_SIZE, 1);
//    dim3 grid(GRID_MAX, div_up(voxels_size, GRID_MAX));
    dim3 block(BLOCK_SIZE, 1);
    uint num_blocks = div_up(voxels_size, block.x);
    uint num_x_blocks = std::min((int)GRID_MAX, (int)num_blocks);
    uint num_y_blocks = div_up(num_blocks, GRID_MAX);
    dim3 grid(num_x_blocks, num_y_blocks);

#ifdef DEBUG_ONE_BLOCK
    grid.x = grid.y = 1;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 512);
#endif
    std::cout << "CONFIG: " << voxels_size << " voxels => " << grid.x << "x" << grid.y << " blocks" << std::endl;

    float3 *timings_d = NULL;
#ifdef TIMINGS
    float3 timings_h = make_float3(0.0f, 0.0f, 0.0f);
    CS(cudaMalloc(&timings_d, sizeof(float3)));
    CS(cudaMemcpy(timings_d, &timings_h, sizeof(float3), cudaMemcpyHostToDevice));
#endif
    ifn_with_octree_1VPT_cuda<<<grid, block>>>(root_center, root_size, nodes, nodes_size, samples, samples_size, voxel_positions, voxels_size, result, timings_d);
    CS(cudaPeekAtLastError());
    CS(cudaDeviceSynchronize());

#ifdef TIMINGS
    CS(cudaMemcpy(&timings_h, timings_d, sizeof(float3), cudaMemcpyDeviceToHost));
    std::cout << "Tree traversal: " << timings_h.x * 100 << "%" << std::endl
#ifdef SCALE_SELECTION
              << "Scale selection: " << timings_h.y * 100 << "%" << std::endl
#endif
              << "Implicit function: " << timings_h.z * 100 << "%" << std::endl;
#endif
}

FSSR_NAMESPACE_END

