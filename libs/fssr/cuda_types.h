#ifndef CUDA_TYPES_HEADER
#define CUDA_TYPES_HEADER

#include <cuda_runtime.h>

namespace CUTypes {

    /* Tree structs */
    struct __align__(16) Node
    {
        uint node_count;
        uint sample_offset;
        uint sample_length;
        uint dummy0;

        __host__ Node(uint _sample_offset, uint _sample_length) : node_count(1), sample_offset(_sample_offset), sample_length(_sample_length) {}
        __host__ Node() {}
    };

    struct __align__(16) Sample
    {
        float3 pos;
        float3 normal;
        float3 color;
        float scale;
        float confidence;
        float dummy0;
    };

    struct __align__(16) VoxelData
    {
        float value;
        float conf;
        float scale;
        float3 color;
        float dummy0, dummy1;
    };

    struct __align__(16) VoxelDataExtended
    {
        float value;
        float conf;
        float scale;
        float3 color;
        float color_weight;
        float dummy0;
    };
}

#endif /* CUDA_TYPES_HEADER */
