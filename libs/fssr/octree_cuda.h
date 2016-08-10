#ifndef FSSR_OCTREE_CUDA_HEADER
#define FSSR_OCTREE_CUDA_HEADER

#include <cuda_runtime.h>

#include "fssr/iso_octree.h"
#include "fssr/cuda_types.h"

FSSR_NAMESPACE_BEGIN


void compute_all_voxels_v2(IsoOctree *octree); // TODO: move anywhere else

class OctreeCUDA
{

public:
    IsoOctree &original;

    float3 root_center;
    float root_size;
    std::vector<CUTypes::Node> nodes;
    std::vector<CUTypes::Sample> samples;
    std::vector<float3> voxel_positions;
    std::vector<CUTypes::VoxelData> result;

    uint nodes_size;
    CUTypes::Node *nodes_d;
    uint samples_size;
    CUTypes::Sample *samples_d;
    uint voxels_size;
    float3 *voxel_positions_d;
    CUTypes::VoxelData *result_d;

public:
    OctreeCUDA( fssr::IsoOctree& _original );

    void
    upload();

    void
    download();

    void
    operator()();

    void
    convert();

private:
    uint
    build_recurse(IsoOctree::Node const& node);

    void
    compute_voxel_positions();

    CUTypes::Sample
    sample2cuda(const Sample &src);

    VoxelData
    cuda2voxeldata(const CUTypes::VoxelData &src);
};


FSSR_NAMESPACE_END

#endif /* FSSR_OCTREE_CUDA_HEADER */
