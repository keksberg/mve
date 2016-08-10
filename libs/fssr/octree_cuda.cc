#include "octree_cuda.h"
#include "sample.h"
#include "ifn_with_octree.h"
#include <set>
#include <list>
#include <iostream>

FSSR_NAMESPACE_BEGIN

void compute_all_voxels_v2(IsoOctree *octree)
{
    OctreeCUDA ot(*octree);
    ot.upload();
    ot();
    ot.download();
    ot.convert();
}

CUTypes::Sample
OctreeCUDA::sample2cuda(const Sample &src)
{
    CUTypes::Sample dst;

    dst.pos = make_float3(src.pos[0], src.pos[1], src.pos[2]);
    dst.normal = make_float3(src.normal[0], src.normal[1], src.normal[2]);
    dst.color = make_float3(src.color[0], src.color[1], src.color[2]);
    dst.scale = src.scale;
    dst.confidence = src.confidence;

    return dst;
}

VoxelData
OctreeCUDA::cuda2voxeldata( const CUTypes::VoxelData& src )
{
    VoxelData dst;

    dst.value = src.value;
    dst.conf = src.conf;
    dst.scale = src.scale;
    dst.color[0] = src.color.x;
    dst.color[1] = src.color.y;
    dst.color[2] = src.color.z;

    return dst;
}

OctreeCUDA::OctreeCUDA(IsoOctree &_original) : original(_original)
{
    std::cout << "Converting input tree..." << std::endl;
    this->build_recurse(*this->original.get_root_node());
    std::cout << "Computing voxel positions..." << std::endl;
    this->compute_voxel_positions();
    math::Vec3d center = this->original.get_root_node_center();
    this->root_center = make_float3(center[0], center[1], center[2]);
    this->root_size = this->original.get_root_node_size();
}

uint OctreeCUDA::build_recurse(IsoOctree::Node const& node)
{
    uint sample_offset = this->samples.size();
    for (uint i = 0; i < node.samples.size(); ++i) {
        this->samples.push_back(this->sample2cuda(node.samples[i]));
    }
    uint sample_length = node.samples.size();

    uint idx = this->nodes.size();
    this->nodes.resize(idx + 1);
    CUTypes::Node parent(sample_offset, sample_length);

    if (node.children != nullptr) {
        for (uint i = 0; i < 8; ++i) {
            IsoOctree::Node const& child = node.children[i];
            parent.node_count += build_recurse(child);
        }
    }

    this->nodes[idx] = parent;

    return parent.node_count;
}

void OctreeCUDA::compute_voxel_positions() {
    /* Locate all leafs and store voxels in a vector. */
    std::cout << "Computing sampling of the implicit function..." << std::endl;
    {
        /* Make voxels unique by storing them in a set first. */
        typedef std::set<VoxelIndex> VoxelIndexSet;
        VoxelIndexSet voxel_set;

        /* Add voxels for all leaf nodes. */
        Octree::Iterator iter = this->original.get_iterator_for_root();
        for (iter.first_leaf(); iter.current != nullptr; iter.next_leaf())
        {
            for (int i = 0; i < 8; ++i)
            {
                VoxelIndex index;
                index.from_path_and_corner(iter.level, iter.path, i);
                voxel_set.insert(index);
            }
        }

        /* Copy voxels over to a vector. */
        this->voxel_positions.reserve(voxel_set.size());
        this->original.voxels.reserve(voxel_set.size());
        for (VoxelIndexSet::const_iterator i = voxel_set.begin(); i != voxel_set.end(); ++i)
        {
            VoxelIndex idx = *i;
            math::Vec3d vp = idx.compute_position(this->original.get_root_node_center(), this->original.get_root_node_size());
            float3 voxel_pos = make_float3(vp[0], vp[1], vp[2]);
            this->voxel_positions.push_back(voxel_pos);

            this->original.voxels.push_back(std::make_pair(*i, VoxelData()));
        }
    }
}

void
print_mem(std::string descr, uint bytes)
{
    std::cout << descr << ": ";
    if (bytes < 1024)
        std::cout << bytes << "B";
    else if (bytes < (1024 << 10))
        std::cout << (bytes >> 10) << "KB";
    else if (bytes < (1024 << 20))
        std::cout << (bytes >> 20) << "MB";
    else
//        std::cout << (bytes >> 30) << "GB";
        std::cout << (bytes >> 20) << "MB";
    std::cout << std::endl;
}

void OctreeCUDA::upload() {
    std::cout << "Uploading..." << std::endl;
    this->nodes_size = this->nodes.size();
    this->samples_size = this->samples.size();
    this->voxels_size = this->voxel_positions.size();

    print_mem("Nodes", this->nodes_size * sizeof(CUTypes::Node));
    CS(cudaMalloc(&this->nodes_d, this->nodes_size * sizeof(CUTypes::Node)));
    print_mem("Samples", this->samples_size * sizeof(CUTypes::Sample));
    CS(cudaMalloc(&this->samples_d, this->samples_size * sizeof(CUTypes::Sample)));
    print_mem("Voxels", this->voxels_size * sizeof(float3));
    CS(cudaMalloc(&this->voxel_positions_d, this->voxels_size * sizeof(float3)));
    print_mem("Result", this->voxels_size * sizeof(CUTypes::VoxelData));
    CS(cudaMalloc(&this->result_d, this->voxels_size * sizeof(CUTypes::VoxelData))); // TODO: Move anywhere else

    CS(cudaMemcpy(this->nodes_d, this->nodes.data(), this->nodes_size * sizeof(CUTypes::Node), cudaMemcpyHostToDevice));
    CS(cudaMemcpy(this->samples_d, this->samples.data(), this->samples_size * sizeof(CUTypes::Sample), cudaMemcpyHostToDevice));
    CS(cudaMemcpy(this->voxel_positions_d, this->voxel_positions.data(), this->voxels_size * sizeof(float3), cudaMemcpyHostToDevice));

    this->nodes.clear();
    this->samples.clear();
    this->voxel_positions.clear();
}

void OctreeCUDA::download() {
    std::cout << "Downloading..." << std::endl;
    this->result.resize(this->voxels_size);
    CS(cudaMemcpy(this->result.data(), this->result_d, this->voxels_size * sizeof(CUTypes::VoxelData), cudaMemcpyDeviceToHost));

    CS(cudaFree(this->nodes_d));
    CS(cudaFree(this->samples_d));
    CS(cudaFree(this->voxel_positions_d));
    CS(cudaFree(this->result_d));
}

void OctreeCUDA::operator()() {
    std::cout << "Calling the kernel..." << std::endl;
    ifn_with_octree_gpu(this->root_center, this->root_size, this->nodes_d, this->nodes_size, this->samples_d, this->samples_size, this->voxel_positions_d, this->voxels_size, this->result_d);
}

void OctreeCUDA::convert() {
    std::cout << "Converting the output data..." << std::endl;
    /* Read back ifn evaluations to voxels. */
    for (std::size_t i = 0; i < this->voxels_size; ++i)
    {
        this->original.voxels[i].second = this->cuda2voxeldata(this->result[i]);
    }
    this->result.clear();
    std::cout << "Finished converting." << std::endl;
}

FSSR_NAMESPACE_END

#if 0
#define DEBUG
#ifdef DEBUG

void fillwithnull(fssr::IsoOctree::Node *node) {
    for (int i = 0; i < 8; ++i) {
        node->children[i] = NULL;
    }
}

int main()
{
    fssr::IsoOctree::Node root;
    fillwithnull(&root);
    for (int i = 0; i < 8; ++i)
    {
        fssr::IsoOctree::Node *child = new fssr::IsoOctree::Node();
        root.children[i] = child;
        fillwithnull(child);

        if (i == 2) {
            for (int j = 0; j < 8; ++j) {
                fssr::IsoOctree::Node *child2 = new fssr::IsoOctree::Node();
                fillwithnull(child2);
                child->children[j] = child2;
            }
        }
    }

    fssr::OctreeCUDA cuda(root);
    for (uint i = 0; i < cuda.nodes.size(); ++i) {
        std::cout << cuda.nodes[i].node_count << ", ";
    }

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < 8; ++i) {
        if (i == 2) {
            for (int j = 0; j < 8; ++j) {
                delete root.children[i]->children[j];
            }
        }
        delete root.children[i];
    }
}

#endif
#endif
