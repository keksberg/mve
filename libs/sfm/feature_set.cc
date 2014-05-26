#include <iostream>
#include <algorithm>
#include <random>

#include "sfm/feature_set.h"

SFM_NAMESPACE_BEGIN

namespace
{
    void
    discretize_descriptor (Sift::Descriptor const& descr,
        math::Vector<unsigned char, 128>* data)
    {
        for (int i = 0; i < 128; ++i)
        {
            float value = descr.data[i];
            value = math::clamp(value, 0.0f, 1.0f);
            value = math::round(value * 255.0f);
            (*data)[i] = static_cast<unsigned char>(value);
        }
    }

    void
    discretize_descriptor (Surf::Descriptor const& descr,
        math::Vector<signed char, 64>* data)
    {
        for (int i = 0; i < 64; ++i)
        {
            float value = descr.data[i];
            value = math::clamp(value, 0.0f, 1.0f);
            value = math::round(value * 255.0f);
            (*data)[i] = static_cast<signed char>(value);
        }
    }
}  /* namespace */

void
FeatureSet::compute_features (mve::ByteImage::Ptr image, float noise)
{
    this->colors.clear();
    this->positions.clear();
    this->width = image->width();
    this->height = image->height();

    /* Make sure these are in the right order. Matching relies on it. */
    if (this->opts.feature_types & FEATURE_SIFT)
        this->compute_sift(image, noise);
    if (this->opts.feature_types & FEATURE_SURF)
        this->compute_surf(image, noise);
}

math::Vec2f
FeatureSet::calc_noise (float noise_stddev)
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, noise_stddev);
    return math::Vec2f(distribution(generator), distribution(generator));
}

void
FeatureSet::compute_sift (mve::ByteImage::ConstPtr image, float noise_stddev)
{
    /* Compute features. */
    Sift sift(this->opts.sift_opts);
    sift.set_image(image);
    sift.process();
    Sift::Descriptors const& descr = sift.get_descriptors();

    /* Prepare and copy to data structures. */
    std::size_t offset = this->positions.size();
    this->positions.resize(offset + descr.size());
    this->colors.resize(offset + descr.size());
    this->sift_descr.allocate(descr.size() * 128);
    this->num_sift_descriptors = descr.size();

    float* ptr = this->sift_descr.begin();
    for (std::size_t i = 0; i < descr.size(); ++i, ptr += 128)
    {
        Sift::Descriptor const& d = descr[i];

        std::copy(d.data.begin(), d.data.end(), ptr);
        this->positions[offset + i] = math::Vec2f(d.x, d.y) + calc_noise(noise_stddev);
        image->linear_at(d.x, d.y, this->colors[offset + i].begin());
    }
}

void
FeatureSet::compute_surf (mve::ByteImage::ConstPtr image, float noise_stddev)
{
    /* Compute features. */
    Surf surf(this->opts.surf_opts);
    surf.set_image(image);
    surf.process();
    Surf::Descriptors const& descr = surf.get_descriptors();

    /* Prepare and copy to data structures. */
    std::size_t offset = this->positions.size();
    this->positions.resize(offset + descr.size());
    this->colors.resize(offset + descr.size());
    this->surf_descr.allocate(descr.size() * 64);
    this->num_surf_descriptors = descr.size();

    float* ptr = this->surf_descr.begin();
    for (std::size_t i = 0; i < descr.size(); ++i, ptr += 64)
    {
        Surf::Descriptor const& d = descr[i];

        std::copy(d.data.begin(), d.data.end(), ptr);
        this->positions[offset + i] = math::Vec2f(d.x, d.y) + calc_noise(noise_stddev);
        image->linear_at(d.x, d.y, this->colors[offset + i].begin());
    }
}

void
FeatureSet::match (FeatureSet const& other, Matching::Result* result) const
{
    /* SIFT matching. */
    sfm::Matching::Result sift_result;
    if (this->num_sift_descriptors > 0)
    {
        sfm::Matching::twoway_match(this->opts.sift_matching_opts,
            this->sift_descr.begin(), this->num_sift_descriptors,
            other.sift_descr.begin(), other.num_sift_descriptors,
            &sift_result);
        sfm::Matching::remove_inconsistent_matches(&sift_result);
    }

    /* SURF matching. */
    sfm::Matching::Result surf_result;
    if (this->num_surf_descriptors > 0)
    {
        sfm::Matching::twoway_match(this->opts.surf_matching_opts,
            this->surf_descr.begin(), this->num_surf_descriptors,
            other.surf_descr.begin(), other.num_surf_descriptors,
            &surf_result);
        sfm::Matching::remove_inconsistent_matches(&surf_result);
    }

    /* Fix offsets in the matching result. */
    int other_surf_offset = other.num_sift_descriptors;
    if (other_surf_offset > 0)
        for (std::size_t i = 0; i < surf_result.matches_1_2.size(); ++i)
            if (surf_result.matches_1_2[i] >= 0)
                surf_result.matches_1_2[i] += other_surf_offset;

    int this_surf_offset = this->num_sift_descriptors;
    if (this_surf_offset > 0)
        for (std::size_t i = 0; i < surf_result.matches_2_1.size(); ++i)
            if (surf_result.matches_2_1[i] >= 0)
                surf_result.matches_2_1[i] += this_surf_offset;

    /* Create a combined matching result. */
    std::size_t this_num_descriptors = this->num_sift_descriptors
        + this->num_surf_descriptors;
    std::size_t other_num_descriptors = other.num_sift_descriptors
        + other.num_surf_descriptors;

    result->matches_1_2.clear();
    result->matches_1_2.reserve(this_num_descriptors);
    result->matches_1_2.insert(result->matches_1_2.end(),
        sift_result.matches_1_2.begin(), sift_result.matches_1_2.end());
    result->matches_1_2.insert(result->matches_1_2.end(),
        surf_result.matches_1_2.begin(), surf_result.matches_1_2.end());

    result->matches_2_1.clear();
    result->matches_2_1.reserve(other_num_descriptors);
    result->matches_2_1.insert(result->matches_2_1.end(),
        sift_result.matches_2_1.begin(), sift_result.matches_2_1.end());
    result->matches_2_1.insert(result->matches_2_1.end(),
        surf_result.matches_2_1.begin(), surf_result.matches_2_1.end());
}

void
FeatureSet::clean_descriptors (void)
{
    this->num_sift_descriptors = 0;
    this->sift_descr.deallocate();
    this->num_surf_descriptors = 0;
    this->surf_descr.deallocate();
}

SFM_NAMESPACE_END
