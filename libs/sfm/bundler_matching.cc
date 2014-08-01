#include <iostream>
#include <fstream>
#include <cstring>
#include <cerrno>

#include "util/exception.h"
#include "util/timer.h"
#include "sfm/sift.h"
#include "sfm/ransac.h"
#include "sfm/bundler_matching.h"


SFM_NAMESPACE_BEGIN
SFM_BUNDLER_NAMESPACE_BEGIN

void
Matching::compute (ViewportList const& viewports,
    PairwiseMatching* pairwise_matching)
{
    std::size_t num_pairs = 0;
    if (this->opts.match_num_previous_frames > 0)
        num_pairs = viewports.size() * this->opts.match_num_previous_frames -
            (this->opts.match_num_previous_frames - 1) *
            (this->opts.match_num_previous_frames - 2) / 2;
    else
        num_pairs = viewports.size() * (viewports.size() - 1) / 2;

    std::size_t current_pair = 0;

    if (this->progress != NULL)
    {
        this->progress->num_total = num_pairs;
        this->progress->num_done = 0;
    }

    pairwise_matching->clear();
    for (std::size_t i = 0; i < viewports.size(); ++i)
    {
        std::size_t start = 0;
        if (this->opts.match_num_previous_frames > 0)
            start = static_cast<int>(i) > this->opts.match_num_previous_frames
                ? i - this->opts.match_num_previous_frames
                : 0;

        for (std::size_t j = start; j < i; ++j)
        {
            current_pair += 1;
            if (this->progress != NULL)
                this->progress->num_done += 1;

            FeatureSet const& view_1 = viewports[i].features;
            FeatureSet const& view_2 = viewports[j].features;
            if (view_1.positions.empty() || view_2.positions.empty())
                continue;

            /* Debug output. */
            int percent = current_pair * 100 / num_pairs;
            std::cout << "Processing pair " << i << ","
                << j << " (" << percent << "%)..." << std::endl;

            /* Match the views. */
            CorrespondenceIndices matches;
            this->two_view_matching(view_1, view_2, &matches);
            if (matches.empty())
                continue;

            /* Successful two view matching. Add the pair. */
            pairwise_matching->push_back(TwoViewMatching());
            TwoViewMatching& matching = pairwise_matching->back();
            matching.view_1_id = i;
            matching.view_2_id = j;
            std::swap(matching.matches, matches);
        }
    }

    std::cout << "Found a total of " << pairwise_matching->size()
        << " matching image pairs." << std::endl;
}

void
Matching::compute_additional (ViewportList const& viewports,
    std::vector<std::vector<std::size_t> > additional_matches,
    PairwiseMatching* pairwise_matching)
{
    if (viewports.size() != additional_matches.size())
        throw std::invalid_argument("Arguments have to have the same size!");

    std::size_t num_pairs = 0;
    for (std::size_t i = 0; i < additional_matches.size(); ++i)
        num_pairs += additional_matches[i].size();
    std::size_t current_pair = 0;

    if (this->progress != NULL)
    {
        this->progress->num_total = num_pairs;
        this->progress->num_done = 0;
    }

    for (std::size_t i = 0; i < additional_matches.size(); ++i)
    {
        for (std::size_t j = 0; j < additional_matches[i].size(); ++j)
        {
            std::size_t idx = additional_matches[i][j];
            if (idx > i)
            {
                additional_matches[idx].push_back(i);
                continue;
            }

            current_pair += 1;
            if (this->progress != NULL)
                this->progress->num_done += 1;

            if (idx == i || idx >= (i - this->opts.match_num_previous_frames))
                continue;

            FeatureSet const& view_1 = viewports[i].features;
            FeatureSet const& view_2 = viewports[idx].features;
            if (view_1.positions.empty() || view_2.positions.empty())
                continue;

            /* Debug output. */
            int percent = current_pair * 100 / num_pairs;
            std::cout << "Processing pair " << i << ","
                << idx << " (" << percent << "%)..." << std::endl;

            /* Match the views. */
            CorrespondenceIndices matches;
            this->two_view_matching(view_1, view_2, &matches);
            if (matches.empty())
                continue;

            /* Successful two view matching. Add the pair. */
            pairwise_matching->push_back(TwoViewMatching());
            TwoViewMatching& matching = pairwise_matching->back();
            matching.view_1_id = i;
            matching.view_2_id = idx;
            std::swap(matching.matches, matches);
        }
    }

    std::cout << "Found a total of " << pairwise_matching->size()
        << " matching image pairs." << std::endl;
}

void
Matching::two_view_matching (FeatureSet const& view_1,
    FeatureSet const& view_2, CorrespondenceIndices* matches)
{
    /* Low-res matching if number of features is large. */
    if (this->opts.use_lowres_matching
        && view_1.positions.size() * view_2.positions.size() > 1000000)
    {
        int const num_matches = view_1.match_lowres(view_2,
            this->opts.num_lowres_features);
        if (num_matches < this->opts.min_lowres_matches)
        {
            std::cout << "  Only " << num_matches
                << " of " << this->opts.min_lowres_matches
                << " low-res matches. Rejecting pair." << std::endl;
            return;
        }
    }

    /* Perform two-view descriptor matching. */
    sfm::Matching::Result matching_result;
    int num_matches = 0;
    {
        util::WallTimer timer;
        view_1.match(view_2, &matching_result);
        num_matches = sfm::Matching::count_consistent_matches(matching_result);
        std::cout << "  Matching took " << timer.get_elapsed() << "ms, "
            << num_matches << " matches." << std::endl;
    }

    /* Require at least 8 matches. Check threshold. */
    int const min_matches_thres = std::max(8, this->opts.min_feature_matches);
    if (num_matches < min_matches_thres)
    {
        std::cout << "  Matches below threshold of "
            << min_matches_thres << ". Skipping." << std::endl;
        return;
    }

    /* Build correspondences from feature matching result. */
    sfm::Correspondences unfiltered_matches;
    CorrespondenceIndices unfiltered_indices;
    {
        std::vector<int> const& m12 = matching_result.matches_1_2;
        for (std::size_t i = 0; i < m12.size(); ++i)
        {
            if (m12[i] < 0)
                continue;

            sfm::Correspondence match;
            match.p1[0] = view_1.positions[i][0];
            match.p1[1] = view_1.positions[i][1];
            match.p2[0] = view_2.positions[m12[i]][0];
            match.p2[1] = view_2.positions[m12[i]][1];
            unfiltered_matches.push_back(match);
            unfiltered_indices.push_back(std::make_pair(i, m12[i]));
        }
    }

    /* Pose RANSAC. */
    sfm::RansacFundamental::Result ransac_result;
    int num_inliers = 0;
    {
        sfm::RansacFundamental ransac(this->opts.ransac_opts);
        util::WallTimer timer;
        ransac.estimate(unfiltered_matches, &ransac_result);
        num_inliers = ransac_result.inliers.size();
        std::cout << "  RANSAC took " << timer.get_elapsed() << "ms, "
            << ransac_result.inliers.size() << " inliers." << std::endl;
    }

    /* Require at least 8 inlier matches. */
    int const min_inlier_thres = std::max(8, this->opts.min_matching_inliers);
    if (num_inliers < min_inlier_thres)
    {
        std::cout << "  Inliers below threshold of "
            << min_inlier_thres << ". Skipping." << std::endl;
        return;
    }

    /* Create Two-View matching result. */
    matches->clear();
    matches->reserve(num_inliers);
    for (int i = 0; i < num_inliers; ++i)
    {
        int const inlier_id = ransac_result.inliers[i];
        matches->push_back(unfiltered_indices[inlier_id]);
    }
}

SFM_BUNDLER_NAMESPACE_END
SFM_NAMESPACE_END
