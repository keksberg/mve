#ifndef FSSR_IFN_WO_INFLUENCE_HEADER
#define FSSR_IFN_WO_INFLUENCE_HEADER

#include <cuda_runtime.h>

#include "fssr/iso_octree.h"

FSSR_NAMESPACE_BEGIN
void compute_all_voxels_v1 (IsoOctree *octree); 


FSSR_NAMESPACE_END

#endif /* FSSR_IFN_WO_INFLUENCE_HEADER */
