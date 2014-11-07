#include <limits>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "math/matrix_tools.h"
#include "sfm/triangulate.h"
#include "sfm/pba_cpu.h"
#include "sfm/bundler_incremental.h"

#include <omp.h>

SFM_NAMESPACE_BEGIN
SFM_BUNDLER_NAMESPACE_BEGIN

void
Incremental::initialize (ViewportList* viewports, TrackList* tracks)
{
    this->viewports = viewports;
    this->tracks = tracks;

    this->cameras.clear();
    this->cameras.resize(viewports->size());

    /* Set track positions to invalid state. */
    for (std::size_t i = 0; i < tracks->size(); ++i)
    {
        Track& track = tracks->at(i);
        track.invalidate();
    }
}

void
Incremental::set_camera_poses (mve::Scene::Ptr scene, std::string image_name)
{
    if (scene->get_views().size() != this->cameras.size())
        throw std::invalid_argument("Invalid view count.");
    for (std::size_t i = 0; i < this->cameras.size(); ++i)
    {
        // TODO make robust
        mve::View::Ptr view = scene->get_view_by_id(i);
        if (view == NULL)
        {
            std::cout << "view " << i << " does not exist" << std::endl;
            continue;
        }
        mve::CameraInfo const& cam_info = view->get_camera();
        CameraPose & cam_pose = this->cameras[i];
        float width = view->get_image(image_name)->width();
        float height = view->get_image(image_name)->height();
        float maxdim = std::max(width, height);
        math::Matrix3f K;
        cam_info.fill_calibration(K.begin(), width, height);
        std::copy(cam_info.rot, cam_info.rot+9, cam_pose.R.begin());
        std::copy(cam_info.trans, cam_info.trans+3, cam_pose.t.begin());
        std::copy(K.begin(), K.end(), cam_pose.K.begin());
        sfm::bundler::Viewport & vp = this->viewports->at(i);
        vp.focal_length = cam_info.flen;
        vp.radial_distortion = cam_info.dist[0] / MATH_POW2(cam_info.flen);
    }
}

bool
Incremental::is_initialized (void) const
{
    return !this->viewports->empty() && !this->tracks->empty();
}

void
Incremental::reconstruct_initial_pair (int view_1_id, int view_2_id)
{
    Viewport const& view_1 = this->viewports->at(view_1_id);
    Viewport const& view_2 = this->viewports->at(view_2_id);

    if (this->opts.verbose_output)
    {
        std::cout << "Computing fundamental matrix for "
            << "initial pair..." << std::endl;
    }

    /* Find the set of fundamental inliers. */
    Correspondences inliers;
    {
        /* Interate all tracks and find correspondences between the pair. */
        Correspondences correspondences;
        for (std::size_t i = 0; i < this->tracks->size(); ++i)
        {
            int view_1_feature_id = -1;
            int view_2_feature_id = -1;
            Track const& track = this->tracks->at(i);
            for (std::size_t j = 0; j < track.features.size(); ++j)
            {
                if (track.features[j].view_id == view_1_id)
                    view_1_feature_id = track.features[j].feature_id;
                if (track.features[j].view_id == view_2_id)
                    view_2_feature_id = track.features[j].feature_id;
            }

            if (view_1_feature_id != -1 && view_2_feature_id != -1)
            {
                math::Vec2f const& pos1 = view_1.features.positions[view_1_feature_id];
                math::Vec2f const& pos2 = view_2.features.positions[view_2_feature_id];
                Correspondence correspondence;
                std::copy(pos1.begin(), pos1.end(), correspondence.p1);
                std::copy(pos2.begin(), pos2.end(), correspondence.p2);
                correspondences.push_back(correspondence);
            }
        }

        /* Use correspondences and compute fundamental matrix using RANSAC. */
        RansacFundamental::Result ransac_result;
        RansacFundamental fundamental_ransac(this->opts.fundamental_opts);
        fundamental_ransac.estimate(correspondences, &ransac_result);

        if (this->opts.verbose_output)
        {
            float num_matches = correspondences.size();
            float num_inliers = ransac_result.inliers.size();
            float percentage = num_inliers / num_matches;

            std::cout << "Pair "
                << "(" << view_1_id << "," << view_2_id << "): "
                << num_matches << " tracks, "
                << num_inliers << " fundamental inliers ("
                << util::string::get_fixed(100.0f * percentage, 2)
                << "%)." << std::endl;
        }

        /* Build correspondences from inliers only. */
        std::size_t const num_inliers = ransac_result.inliers.size();
        inliers.resize(num_inliers);
        for (std::size_t i = 0; i < num_inliers; ++i)
            inliers[i] = correspondences[ransac_result.inliers[i]];
    }

    /* Save a test match for later to resolve camera pose ambiguity. */
    Correspondence test_match = inliers[0];

    /* Normalize inliers and re-compute fundamental. */
    FundamentalMatrix fundamental;
    {
        math::Matrix3d T1, T2;
        FundamentalMatrix F;
        compute_normalization(inliers, &T1, &T2);
        apply_normalization(T1, T2, &inliers);
        fundamental_least_squares(inliers, &F);
        enforce_fundamental_constraints(&F);
        fundamental = T2.transposed() * F * T1;
        inliers.clear();
    }

    if (this->opts.verbose_output)
    {
        std::cout << "Extracting pose for initial pair..." << std::endl;
    }

    /* Compute pose from fundamental matrix. */
    CameraPose pose1, pose2;
    {
        /* Populate K-matrices. */
        int const width1 = view_1.width;
        int const height1 = view_1.height;
        int const width2 = view_2.width;
        int const height2 = view_2.height;
        double flen1 = view_1.focal_length
            * static_cast<double>(std::max(width1, height1));
        double flen2 = view_2.focal_length
            * static_cast<double>(std::max(width2, height2));
        pose1.set_k_matrix(flen1, width1 / 2.0, height1 / 2.0);
        pose1.init_canonical_form();
        pose2.set_k_matrix(flen2, width2 / 2.0, height2 / 2.0);

        /* Compute essential matrix from fundamental matrix (HZ (9.12)). */
        EssentialMatrix E = pose2.K.transposed() * fundamental * pose1.K;

        /* Compute pose from essential. */
        std::vector<CameraPose> poses;
        pose_from_essential(E, &poses);

        /* Find the correct pose using point test (HZ Fig. 9.12). */
        bool found_pose = false;
        for (std::size_t i = 0; i < poses.size(); ++i)
        {
            poses[i].K = pose2.K;
            if (is_consistent_pose(test_match, pose1, poses[i]))
            {
                pose2 = poses[i];
                found_pose = true;
                break;
            }
        }

        if (!found_pose)
            throw std::runtime_error("Could not find valid initial pose");
    }

    /* Store recovered pose in viewport. */
    this->cameras[view_1_id] = pose1;
    this->cameras[view_2_id] = pose2;
}

/* ---------------------------------------------------------------- */

int
Incremental::find_next_view (void) const
{
    /*
     * The next view is selected by finding the unreconstructed view with
     * most reconstructed tracks.
     */
    std::vector<int> valid_tracks_counter(this->cameras.size(), 0);
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        Track const& track = this->tracks->at(i);
        if (!track.is_valid())
            continue;

        for (std::size_t j = 0; j < track.features.size(); ++j)
        {
            int const view_id = track.features[j].view_id;
            if (this->cameras[view_id].is_valid())
                continue;
            valid_tracks_counter[view_id] += 1;
        }
    }

    std::size_t next_view = math::algo::max_element_id
        (valid_tracks_counter.begin(), valid_tracks_counter.end());

    return valid_tracks_counter[next_view] > 6 ? next_view : -1;
}

/* ---------------------------------------------------------------- */

void
Incremental::find_next_views (std::vector<int>* next_views)
{
    std::vector<std::pair<int, int> > valid_tracks(this->cameras.size());

    for (std::size_t i = 0; i < valid_tracks.size(); ++i)
        valid_tracks[i] = std::pair<int, int>(0, static_cast<int>(i));

    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        Track const& track = this->tracks->at(i);
        if (!track.is_valid())
            continue;

        for (std::size_t j = 0; j < track.features.size(); ++j)
        {
            int const view_id = track.features[j].view_id;
            if (this->cameras[view_id].is_valid())
                continue;
            valid_tracks[view_id].first += 1;
        }
    }

    std::sort(valid_tracks.rbegin(), valid_tracks.rend());

    next_views->clear();
    for (std::size_t i = 0; i < valid_tracks.size(); ++i)
    {
        if (valid_tracks[i].first > 6)
            next_views->push_back(valid_tracks[i].second);
    }
}

/* ---------------------------------------------------------------- */

bool Incremental::reconstruct_next_view (int view_id)
{
    Viewport const& viewport = this->viewports->at(view_id);
    FeatureSet const& features = viewport.features;

    alternate_camera.K.fill(0.0);

    /* Collect all 2D-3D correspondences. */
    Correspondences2D3D corr;
    std::vector<int> track_ids;
    std::vector<int> feature_ids;
    for (std::size_t i = 0; i < viewport.track_ids.size(); ++i)
    {
        int const track_id = viewport.track_ids[i];
        if (track_id < 0 || !this->tracks->at(track_id).is_valid())
            continue;
        math::Vec2f const& pos2d = features.positions[i];
        math::Vec3f const& pos3d = this->tracks->at(track_id).pos;

        corr.push_back(Correspondence2D3D());
        Correspondence2D3D& c = corr.back();
        std::copy(pos3d.begin(), pos3d.end(), c.p3d);
        std::copy(pos2d.begin(), pos2d.end(), c.p2d);
        track_ids.push_back(track_id);
        feature_ids.push_back(i);
    }

    if (this->opts.verbose_output)
    {
        std::cout << "Collected " << corr.size()
            << " 2D-3D correspondences." << std::endl;
    }

    /*
     * Given correspondences, use either P3P or 6-point algorithm.
     * 6-point, delete tracks threshold 10: 24134 features
     * 3-point, delete tracks threshold 10: 25828 features
     * 6-point, delete tracks threshold 20: 41018 features
     * 3-point, delete tracks threshold 20: 42048 features
     */
#define USE_P3P_FOR_POSE 1

    /* Initialize a temporary camera. */
    float const maxdim = static_cast<float>
        (std::max(viewport.width, viewport.height));
    CameraPose temp_camera;
    temp_camera.set_k_matrix(viewport.focal_length * maxdim,
        static_cast<float>(viewport.width) / 2.0f,
        static_cast<float>(viewport.height) / 2.0f);

#if USE_P3P_FOR_POSE
    if (corr.size() < 3)
        return false;

    /* Compute pose from 2D-3D correspondences using P3P. */
    RansacPoseP3P ransac(this->opts.pose_p3p_opts);
    RansacPoseP3P::Result ransac_result;
    ransac.estimate(corr, temp_camera.K, &ransac_result);

    if (this->opts.verbose_output)
    {
        std::cout << "Selected " << ransac_result.inliers.size()
            << " 2D-3D correspondences inliers." << std::endl;
    }

#else
    if (corr.size() < 6)
        return false;

    /* Compute pose from 2D-3D correspondences using 6-point. */
    RansacPose ransac(this->opts.pose_opts);
    RansacPose::Result ransac_result;
    ransac.estimate(corr, &ransac_result);

    if (this->opts.verbose_output)
    {
        std::cout << "RANSAC found " << ransac_result.inliers.size()
            << " inliers." << std::endl;
    }
#endif

    int const ratio = 3; // inliers have to be more than 1/2
    /* Cancel if inliers are below a threshold. */
    if (ratio * ransac_result.inliers.size() < corr.size())
        return false;

    /* P3P: second run... */
    Correspondences2D3D new_corr;
    for (std::size_t i = 0; i < corr.size(); ++i)
    {
        std::vector<int> const& v = ransac_result.inliers;
        if (std::find(v.begin(), v.end(), i) != v.end())
            continue;
        new_corr.push_back(corr[i]);
    }
    RansacPoseP3P::Result ransac_2nd_result;
    if (new_corr.size() > 5)
    {
        RansacPoseP3P ransac_2nd(this->opts.pose_p3p_opts);
        ransac_2nd.estimate(new_corr, temp_camera.K, &ransac_2nd_result);
        std::string debug_filename = "ransac.txt";
        std::ofstream debug_file(debug_filename.c_str(), std::ofstream::app);
        debug_file << view_id << "," << ransac_result.inliers.size() << ","
                   << corr.size() << "," << ransac_2nd_result.inliers.size()
                   << "," << new_corr.size() << std::endl;
        debug_file.close();
        if (ratio * ransac_2nd_result.inliers.size() >= new_corr.size())
        {
            std::cout << "Second RANSAC was successful for view "
                      << view_id << "." << std::endl;
            alternate_camera = temp_camera;
            alternate_camera.R = ransac_2nd_result.pose.delete_col(3);
            alternate_camera.t = ransac_2nd_result.pose.col(3);
            std::string debug_fn = "ransac_info.txt";
            std::ofstream info(debug_fn.c_str(), std::ofstream::app);
            math::Vec3d const& t1 = ransac_result.pose.col(3);
            math::Vec3d const& t2 = alternate_camera.t;
            info << view_id << ","
                 << t1[0] << "," << t1[1] << "," << t1[2] << ","
                 << t2[0] << "," << t2[1] << "," << t2[2] << std::endl;
        }
    }

#if USE_P3P_FOR_POSE
    /* In the P3P case, just use the known K and computed R and t. */
    this->cameras[view_id] = temp_camera;
    this->cameras[view_id].R = ransac_result.pose.delete_col(3);
    this->cameras[view_id].t = ransac_result.pose.col(3);
#else
    /* With 6-point, set full pose recovering R and t using known K. */
    math::Matrix<double, 3, 4> p_matrix = ransac_result.p_matrix;
    this->cameras[view_id] = temp_camera;
    this->cameras[view_id].set_from_p_and_known_k(p_matrix);
#endif

//    /* Compute and dump reprojection error of the track. */
//    std::string filename = "reprojection.csv";
//    std::ofstream reprojection(filename.c_str(), std::ofstream::app);
//    for (std::size_t j = 0; j < track_ids.size(); ++j)
//    {
//        Correspondence2D3D const& c = corr[j];
//        math::Vec3d p3d(&c.p3d[0]);
//        math::Vec2d p2d(&c.p2d[0]);
//        math::Vec3d x = this->cameras[view_id].R *
//            p3d + this->cameras[view_id].t;
//        x = this->cameras[view_id].K * x;
//        math::Vec2d x2d(x[0] / x[2], x[1] / x[2]);
//        double square_error = (p2d - x2d).square_norm();
//        if (j == track_ids.size() - 1)
//            reprojection << square_error << std::endl;
//        else
//            reprojection << square_error << ";";
//    }
//    reprojection.close();

    /* Remove outliers from tracks and tracks from viewport. */
    int removed_outliers = 0;
    for (std::size_t i = 0; i < ransac_result.inliers.size(); ++i)
        track_ids[ransac_result.inliers[i]] = -1;
//    for (std::size_t i = 0; i < ransac_2nd_result.inliers.size(); ++i)
//        track_ids[ransac_2nd_result.inliers[i]] = -1;
    for (std::size_t i = 0; i < track_ids.size(); ++i)
    {
        if (track_ids[i] < 0)
            continue;
        this->tracks->at(track_ids[i]).remove_view(view_id);
        this->viewports->at(view_id).track_ids[feature_ids[i]] = -1;
        removed_outliers += 1;
    }
    track_ids.clear();
    feature_ids.clear();

    if (this->opts.verbose_output)
    {
        std::cout << "Reconstructed new camera with focal length: "
            << this->cameras[view_id].get_focal_length() << std::endl;
    }

    return true;
}

/* ---------------------------------------------------------------- */

void
Incremental::triangulate_new_tracks (void)
{
    /* Thresholds. */
    double const square_thres = MATH_POW2(this->opts.new_track_error_threshold);
    double const cos_angle_thres = std::cos(this->opts.min_triangulation_angle);

    /* Statistics. */
    int num_new_tracks = 0;
    int num_large_error_tracks = 0;
    int num_behind_camera_tracks = 0;
    int num_too_small_angle = 0;

    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        /* Skip tracks that have already been reconstructed. */
        Track& track = this->tracks->at(i);
        if (track.is_valid())
            continue;

        /*
         * Triangulate a new track using all cameras.
         * There can be more than two cameras if the track has been rejected
         * in previous attempts to triangulate the track.
         */
        std::vector<math::Vec2f> pos;
        std::vector<CameraPose const*> poses;
        for (std::size_t j = 0; j < track.features.size(); ++j)
        {
            int const view_id = track.features[j].view_id;
            if (!this->cameras[view_id].is_valid())
                continue;
            int const feature_id = track.features[j].feature_id;
            pos.push_back(this->viewports->at(view_id)
                .features.positions[feature_id]);
            poses.push_back(&this->cameras[view_id]);
        }

        /* Skip tracks with too few valid cameras. */
        if (poses.size() < 2)
            continue;

        /* Triangulate track. */
        math::Vec3d track_pos = triangulate_track(pos, poses);

        /* Skip tracks with too small triangulation angle. */
        double smallest_cos_angle = 1.0;
        if (this->opts.min_triangulation_angle > 0.0)
        {
            std::vector<math::Vec3d> rays(poses.size());
            for (std::size_t j = 0; j < poses.size(); ++j)
            {
                math::Vec3d camera_pos;
                poses[j]->fill_camera_pos(&camera_pos);
                rays[j] = (track_pos - camera_pos).normalized();
            }

            for (std::size_t j = 0; j < rays.size(); ++j)
                for (std::size_t k = 0; k < j; ++k)
                {
                    double const cos_a = rays[j].dot(rays[k]);
                    smallest_cos_angle = std::min(smallest_cos_angle, cos_a);
                }
            if (smallest_cos_angle > cos_angle_thres)
            {
                num_too_small_angle += 1;
                continue;
            }
        }

        /* Compute reprojection error of the track. */
        double square_error = 0.0;
        bool track_behind_camera = false;
        for (std::size_t j = 0; j < poses.size(); ++j)
        {
            math::Vec3d x = poses[j]->R * track_pos + poses[j]->t;
            if (x[2] < 0.0)
            {
                track_behind_camera = true;
                break;
            }

            x = poses[j]->K * x;
            math::Vec2d x2d(x[0] / x[2], x[1] / x[2]);
            square_error += (pos[j] - x2d).square_norm();
        }
        square_error /= static_cast<double>(poses.size());

        /*
         * Reject track if the reprojection error is large, or the track
         * appears behind the camera. In the latter case, delete it.
         */
        if (track_behind_camera)
        {
            num_behind_camera_tracks += 1;
            this->delete_track(i);
            continue;
        }
        else if (square_error > square_thres)
        {
            num_large_error_tracks += 1;
            continue;
        }

        track.pos = track_pos;
        num_new_tracks += 1;
    }

    if (this->opts.verbose_output)
    {
        int num_rejected = num_large_error_tracks
            + num_behind_camera_tracks + num_too_small_angle;
        std::cout << "Triangulated " << num_new_tracks
            << " new tracks, rejected " << num_rejected
            << " bad tracks." << std::endl;
        if (num_large_error_tracks > 0)
            std::cout << "  Rejected " << num_large_error_tracks
                << " tracks with large error." << std::endl;
        if (num_behind_camera_tracks > 0)
            std::cout << "  Rejected " << num_behind_camera_tracks
                << " tracks behind cameras." << std::endl;
        if (num_too_small_angle > 0)
            std::cout << "  Rejected " << num_too_small_angle
                << " tracks with unstable angle." << std::endl;
    }
}

/* ---------------------------------------------------------------- */

void
Incremental::bundle_adjustment_full (void)
{
    if(this->opts.use_ceres_solver)
        this->bundle_adjustment_ceres_intern(-1);
    else
        this->bundle_adjustment_intern(-1);
}

/* ---------------------------------------------------------------- */

void
Incremental::bundle_adjustment_single_cam (int view_id)
{
    if(this->opts.use_ceres_solver)
        this->bundle_adjustment_ceres_intern(view_id);
    else
        this->bundle_adjustment_intern(view_id);
}

/* ---------------------------------------------------------------- */

//#define PBA_DISTORTION_TYPE pba::PROJECTION_DISTORTION
#define PBA_DISTORTION_TYPE pba::MEASUREMENT_DISTORTION
//#define PBA_DISTORTION_TYPE pba::NO_DISTORTION

void
Incremental::bundle_adjustment_intern (int single_camera_ba)
{
    /* Configure PBA. */
    pba::SparseBundleCPU pba;
    pba.EnableRadialDistortion(PBA_DISTORTION_TYPE);
    pba.SetNextTimeBudget(0);
    if (single_camera_ba >= 0)
        pba.SetNextBundleMode(pba::BUNDLE_ONLY_MOTION);
    else
        pba.SetNextBundleMode(pba::BUNDLE_FULL);

    pba.SetNextTimeBudget(0);
    pba.SetFixedIntrinsics(this->opts.ba_fixed_intrinsics);

    pba.GetInternalConfig()->__verbose_cg_iteration = false;
    pba.GetInternalConfig()->__verbose_level = -1;
    pba.GetInternalConfig()->__verbose_function_time = false;
    pba.GetInternalConfig()->__verbose_allocation = false;
    pba.GetInternalConfig()->__verbose_sse = false;
    pba.GetInternalConfig()->__lm_max_iteration = 500;
    //pba.GetInternalConfig()->__cg_min_iteration = 30;
    pba.GetInternalConfig()->__cg_max_iteration = 500;
    //pba.GetInternalConfig()->__lm_delta_threshold = 1E-7;
    //pba.GetInternalConfig()->__lm_mse_threshold = 1E-2;

    /* Prepare camera data. */
    std::vector<pba::CameraT> pba_cams;
    std::vector<int> pba_cams_mapping(this->cameras.size(), -1);
    for (std::size_t i = 0; i < this->cameras.size(); ++i)
    {
        if (!this->cameras[i].is_valid())
            continue;

        CameraPose const& pose = this->cameras[i];
        pba::CameraT cam;
        cam.f = pose.get_focal_length();
        std::copy(pose.t.begin(), pose.t.end(), cam.t);
        //std::cout << "pose.t (" << pose.t TODO FIXME
        std::copy(pose.R.begin(), pose.R.end(), cam.m[0]);
        cam.radial = this->viewports->at(i).radial_distortion;
        cam.distortion_type = PBA_DISTORTION_TYPE;
        pba_cams_mapping[i] = pba_cams.size();

        if (single_camera_ba >= 0 && (int)i != single_camera_ba)
            cam.SetConstantCamera();

        pba_cams.push_back(cam);
    }
    pba.SetCameraData(pba_cams.size(), &pba_cams[0]);

    /* Prepare tracks data. */
    std::vector<pba::Point3D> pba_tracks;
    std::vector<int> pba_tracks_mapping(this->tracks->size(), -1);
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        Track const& track = this->tracks->at(i);
        if (!track.is_valid())
            continue;

        pba::Point3D point;
        std::copy(track.pos.begin(), track.pos.end(), point.xyz);
        pba_tracks_mapping[i] = pba_tracks.size();
        pba_tracks.push_back(point);
    }
    pba.SetPointData(pba_tracks.size(), &pba_tracks[0]);

    /* Prepare feature positions in the images. */
    std::vector<pba::Point2D> pba_2d_points;
    std::vector<int> pba_track_ids;
    std::vector<int> pba_cam_ids;
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        Track const& track = this->tracks->at(i);
        if (!track.is_valid())
            continue;

        for (std::size_t j = 0; j < track.features.size(); ++j)
        {
            int const view_id = track.features[j].view_id;
            if (!this->cameras[view_id].is_valid())
                continue;

            int const feature_id = track.features[j].feature_id;
            Viewport const& view = this->viewports->at(view_id);
            math::Vec2f f2d = view.features.positions[feature_id];

            pba::Point2D point;
            point.x = f2d[0] - static_cast<float>(view.width) / 2.0f;
            point.y = f2d[1] - static_cast<float>(view.height) / 2.0f;

            pba_2d_points.push_back(point);
            pba_track_ids.push_back(pba_tracks_mapping[i]);
            pba_cam_ids.push_back(pba_cams_mapping[view_id]);
        }
    }
    pba.SetProjection(pba_2d_points.size(),
        &pba_2d_points[0], &pba_track_ids[0], &pba_cam_ids[0]);

    /* Run bundle adjustment. */
    pba.RunBundleAdjustment();

    /* Transfer camera info and track positions back. */
    std::size_t pba_cam_counter = 0;
    for (std::size_t i = 0; i < this->cameras.size(); ++i)
    {
        if (!this->cameras[i].is_valid())
            continue;

        CameraPose& pose = this->cameras[i];
        Viewport& view = this->viewports->at(i);
        pba::CameraT const& cam = pba_cams[pba_cam_counter];
        std::copy(cam.t, cam.t + 3, pose.t.begin());
        std::copy(cam.m[0], cam.m[0] + 9, pose.R.begin());

        if (this->opts.verbose_output && single_camera_ba < 0)
        {
            std::cout << "Camera " << i << ", focal length: "
                << pose.get_focal_length() << " -> " << cam.f
                << ", distortion: " << cam.radial << std::endl;
        }

        pose.K[0] = cam.f;
        pose.K[4] = cam.f;
        view.radial_distortion = cam.radial;
        pba_cam_counter += 1;
    }

    std::size_t pba_track_counter = 0;
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        Track& track = this->tracks->at(i);
        if (!track.is_valid())
            continue;

        pba::Point3D const& point = pba_tracks[pba_track_counter];
        std::copy(point.xyz, point.xyz + 3, track.pos.begin());

        pba_track_counter += 1;
    }
}

/* ---------------------------------------------------------------- */

/*
 * The camera is parameterized using 9 parameters: 3 for rotation,
 * 3 for translation, 1 for focal length and 2 for radial distortion.
 */
struct ReprojectionError {
  ReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera_ext,
                  const T* const camera_flen,
                  const T* const camera_radial,
                  const T* const point,
                  T* residuals) const {
    // camera_ext[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera_ext, point, p);

    // translation
    p[0] += camera_ext[3];
    p[1] += camera_ext[4];
    p[2] += camera_ext[5];

    // radial distortion factor
    T rd = T(1.0) + camera_radial[0] * T(observed_x*observed_x + observed_y*observed_y);

    // projection factor
    const T& f_p2 = camera_flen[0] / p[2];

    // The error is the difference between the predicted and observed position.
    residuals[0] = observed_x * rd - p[0] * f_p2;
    residuals[1] = observed_y * rd - p[1] * f_p2;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
      return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 1, 1, 3>(
                new ReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};

#define BA_WEIGHT 1000.0

struct TrackDistanceError {
  template <typename T>
  bool operator()(const T* const point1,
                  const T* const point2,
                  T* residuals) const {

    for (std::size_t i = 0; i < 3; ++i)
        residuals[i] = (point1[i] - point2[i]) * T(100.0 * BA_WEIGHT);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(void) {
      return (new ceres::AutoDiffCostFunction<TrackDistanceError, 3, 3, 3>(
                new TrackDistanceError()));
  }
};

struct CameraDistanceError {
  template <typename T>
  bool operator()(const T* const cam1,
                  const T* const cam2,
                  T* residuals) const {

    for (std::size_t i = 0; i < 3; ++i)
        residuals[i] = (cam1[i+3] - cam2[i+3]) * T(BA_WEIGHT * 10.0);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(void) {
      return (new ceres::AutoDiffCostFunction<CameraDistanceError, 3, 6, 6>(
                new CameraDistanceError()));
  }
};

struct CeresCam
{
    double R[3]; // axis angle rotation
    double t[3];
    double f;
    double radial[2];
    bool fixed;
    double* begin_ext (void)
    {
        return R;
    }
    double* begin_int (void)
    {
        return &f;
    }
    double* begin_flen (void)
    {
        return &f;
    }
    double* begin_radial (void)
    {
        return radial;
    }
};

void
Incremental::bundle_adjustment_ceres_intern (int single_camera_ba)
{
    std::cout << "preparing bundle adjustment ..." << std::endl;

    /* Build bundle adjustment problem. */
    ceres::Problem problem;

    /* Prepare camera data. */
    std::vector<CeresCam> ba_cams;
    std::vector<int> ba_cams_mapping(this->cameras.size(), -1);
    for (std::size_t i = 0; i < this->cameras.size(); ++i)
    {
        if (!this->cameras[i].is_valid())
            continue;

        CameraPose const& pose = this->cameras[i];
        CeresCam cam;
        ceres::RotationMatrixToAngleAxis(
            ceres::RowMajorAdapter3x3(pose.R.begin()), cam.R);
        std::copy(pose.t.begin(), pose.t.end(), cam.t);
        cam.f = pose.get_focal_length();
        cam.radial[0] = this->viewports->at(i).radial_distortion;
        cam.radial[1] = 0.0; // TODO use 2nd coeff.
        cam.fixed = false;

        ba_cams_mapping[i] = ba_cams.size();

        if (single_camera_ba >= 0 && (int)i != single_camera_ba)
            cam.fixed = true;

        ba_cams.push_back(cam);
    }

//    if (this->alternate_camera.is_valid())
//    {
//        CameraPose const& pose = this->alternate_camera;
//        CeresCam cam;
//        ceres::RotationMatrixToAngleAxis(
//            ceres::RowMajorAdapter3x3(pose.R.begin()), cam.R);
//        std::copy(pose.t.begin(), pose.t.end(), cam.t);
//        cam.f = pose.get_focal_length();
//        cam.radial[0] = 0.0;
//        cam.radial[1] = 0.0; // TODO use 2nd coeff.
//        cam.fixed = false;

//        ba_cams_mapping.push_back(ba_cams.size());
//        ba_cams.push_back(cam);
//    }

    /* Prepare tracks data. */
    std::vector<math::Vec3d> ba_tracks;
    std::vector<int> ba_tracks_mapping(this->tracks->size(), -1);
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        Track const& track = this->tracks->at(i);
        if (!track.is_valid())
            continue;

        math::Vec3d point;
        std::copy(track.pos.begin(), track.pos.end(), point.begin());
        ba_tracks_mapping[i] = ba_tracks.size();
        ba_tracks.push_back(point);
    }

    /* Prepare feature positions in the images. (a.k.a. observations) */
    std::vector<math::Vec2d> ba_2d_points;
    std::vector<int> ba_track_ids;
    std::vector<int> ba_cam_ids;
    std::vector<bool> ba_is_lc;
    std::vector<std::pair<int, int> > ba_cam_constraints;
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        std::vector<Track*> multiple_tracks;
        multiple_tracks.push_back(&this->tracks->at(i));

        if (!multiple_tracks[0]->is_valid())
            continue;

        if (this->tracks_matching)
            for (std::size_t j = 0; j < this->tracks_matching->size(); ++j)
                if (this->tracks_matching->at(j).first == i)
                    multiple_tracks.push_back(
                        &this->tracks->at(tracks_matching->at(j).second));

        bool is_closure = multiple_tracks.size() > 1;

        for (std::size_t k = 0; k < multiple_tracks.size(); ++k)
        {
            Track const& track = *multiple_tracks[k];
            for (std::size_t j = 0; j < track.features.size(); ++j)
            {
                int const view_id = track.features[j].view_id;
                if (!this->cameras[view_id].is_valid())
                    continue;

                int const feature_id = track.features[j].feature_id;
                Viewport const& view = this->viewports->at(view_id);
                math::Vec2f f2d = view.features.positions[feature_id];

                math::Vec2d point;
                point[0] = f2d[0] - static_cast<double>(view.width) / 2.0;
                point[1] = f2d[1] - static_cast<double>(view.height) / 2.0;

                // check if point is behind camera
                CameraPose const& pose = this->cameras[view_id];
                math::Vec3d const& point_3d = track.pos;
                math::Vec3d x = (pose.R * point_3d + pose.t);
                if (!is_closure && x[2] < 0.0)
                {
                    //std::cerr << "%% point is behind camera %%" << std::endl;
                    continue;
                }

                ba_2d_points.push_back(point);
                ba_track_ids.push_back(ba_tracks_mapping[i]);
                ba_cam_ids.push_back(ba_cams_mapping[view_id]);
                ba_is_lc.push_back(is_closure);

//                if (this->alternate_camera.is_valid())
//                {
//                    ba_2d_points.push_back(point);
//                    ba_track_ids.push_back(ba_tracks_mapping[i]);
//                    ba_cam_ids.push_back(ba_cams_mapping.back());
//                    ba_is_lc.push_back(is_closure);
//                    ba_cam_constraints.push_back(std::make_pair(
//                        ba_cams_mapping[view_id], ba_cams_mapping.back()));
//                }
            }
        }

    }

    std::cout << "cauchy" << std::endl;
    for (std::size_t i = 0; i < ba_2d_points.size(); ++i)
    {
        ceres::CostFunction* cost_function;
        math::Vec2d const& proj = ba_2d_points[i];
        cost_function = ReprojectionError::Create(proj[0], proj[1]);

        ceres::LossFunction* loss_function = NULL;
        //loss_function = new ceres::HuberLoss(1.0);
        if (ba_is_lc[i])
            loss_function = new ceres::ScaledLoss(
                        new ceres::HuberLoss(1.0),
                        BA_WEIGHT, ceres::DO_NOT_TAKE_OWNERSHIP);
        else
            loss_function = new ceres::CauchyLoss(1.0);

        double* camera_ext = ba_cams[ba_cam_ids[i]].begin_ext();
        //double* camera_int = ba_cams[ba_cam_ids[i]].begin_int();
        double* camera_flen = ba_cams[ba_cam_ids[i]].begin_flen();
        double* camera_radial = ba_cams[ba_cam_ids[i]].begin_radial();
        double* point = ba_tracks[ba_track_ids[i]].begin();

        problem.AddResidualBlock(cost_function, loss_function,
                                 camera_ext, camera_flen,
                                 camera_radial, point);

        // Bundle adjust only motion if single camera requested
        if (single_camera_ba >= 0 || single_camera_ba == -3) // TODO hack
            problem.SetParameterBlockConstant(point);

        if (single_camera_ba == -3)
            problem.SetParameterBlockConstant(camera_flen);

        // Bundle adjust only requested camera
        if (ba_cams[ba_cam_ids[i]].fixed)
        {
            problem.SetParameterBlockConstant(camera_ext);
            problem.SetParameterBlockConstant(camera_flen);
            problem.SetParameterBlockConstant(camera_radial);
        }

        // Set fixed instrinsics if requested
        if ((this->opts.ba_fixed_intrinsics && single_camera_ba != -2) // TODO hack
                || ba_is_lc[i] || single_camera_ba == -4)
        {
            problem.SetParameterBlockConstant(camera_flen);
            problem.SetParameterBlockConstant(camera_radial);
        }
    }

    if (this->tracks_matching)
    {
        std::cout << "Adding loop closing constraints!" << std::endl;
        for (std::size_t i = 0; i < this->tracks_matching->size(); ++i)
        {
            std::size_t first = this->tracks_matching->at(i).first;
            std::size_t second = this->tracks_matching->at(i).second;
            double* point1 = ba_tracks[ba_tracks_mapping[first]].begin();
            double* point2 = ba_tracks[ba_tracks_mapping[second]].begin();
            ceres::CostFunction* cost_function;
            cost_function = TrackDistanceError::Create();

            ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);

            problem.AddResidualBlock(cost_function, loss_function,
                                     point1, point2);
        }
    }

//    for (std::size_t i = 0; i < ba_cam_constraints.size(); ++i)
//    {
//        std::pair<int, int> const& pair = ba_cam_constraints[i];
//        double* camera_ext_1 = ba_cams[pair.first].begin_ext();
//        double* camera_ext_2 = ba_cams[pair.second].begin_ext();
//        ceres::CostFunction* cost_function;
//        cost_function = CameraDistanceError::Create();

//        ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);

//        problem.AddResidualBlock(cost_function, loss_function,
//                                 camera_ext_1, camera_ext_2);
//    }

    /* Set bundle adjustment options (TODO). */
    ceres::Solver::Options options;

    // http://ceres-solver.org/faqs.html#solving
    if (ba_cams.size() < 100)
        options.linear_solver_type = ceres::DENSE_SCHUR;
    else if (ba_cams.size() < 2000)
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    else
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;

    options.gradient_tolerance = 1e-10;
    options.function_tolerance = 1e-6;
    options.max_num_iterations = 500;
    options.max_linear_solver_iterations = 500;

    options.num_threads = omp_get_num_procs();
    options.num_linear_solver_threads = omp_get_num_procs();
    std::cout << "Using " << options.num_threads << " threads." << std::endl;

    std::cout << "running bundle adjustment ..." << std::endl;

    /* Run bundle adjustment. */
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    /* Transfer camera info and track positions back. */
    std::size_t ba_cam_counter = 0;
    for (std::size_t i = 0; i < this->cameras.size(); ++i)
    {
        if (!this->cameras[i].is_valid())
            continue;

        CameraPose& pose = this->cameras[i];
        Viewport& view = this->viewports->at(i);
        CeresCam const& cam = ba_cams[ba_cam_counter];
        std::copy(cam.t, cam.t + 3, pose.t.begin());
        ceres::AngleAxisToRotationMatrix(cam.R,
            ceres::RowMajorAdapter3x3(pose.R.begin()));

        if (this->opts.verbose_output && single_camera_ba < 0)
        {
            std::cout << "Camera " << i << ", focal length: "
                << pose.get_focal_length() << " -> " << cam.f
                << ", distortion: " << cam.radial[0]
                << "," << cam.radial[1] << std::endl;
        }

        pose.K[0] = cam.f;
        pose.K[4] = cam.f;
        view.radial_distortion = cam.radial[0];
        ba_cam_counter += 1;
    }

    std::size_t ba_track_counter = 0;
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        Track& track = this->tracks->at(i);
        if (!track.is_valid())
            continue;

        math::Vec3d const& point = ba_tracks[ba_track_counter];
        std::copy(point.begin(), point.end(), track.pos.begin());

        ba_track_counter += 1;
    }
}

/* ---------------------------------------------------------------- */

void
Incremental::delete_large_error_tracks (void)
{
    /* Iterate over all tracks and sum reprojection error. */
    std::vector<std::pair<double, std::size_t> > all_errors;
    std::size_t num_valid_tracks = 0;
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        if (!this->tracks->at(i).is_valid())
            continue;

        num_valid_tracks += 1;
        math::Vec3f const& pos3d = this->tracks->at(i).pos;
        FeatureReferenceList const& ref = this->tracks->at(i).features;

        double total_error = 0.0f;
        int num_valid = 0;
        for (std::size_t j = 0; j < ref.size(); ++j)
        {
            /* Get pose and 2D position of feature. */
            int view_id = ref[j].view_id;
            int feature_id = ref[j].feature_id;
            CameraPose const& pose = this->cameras[view_id];
            if (!pose.is_valid())
                continue;

            Viewport const& viewport = this->viewports->at(view_id);
            math::Vec2f const& pos2d = viewport.features.positions[feature_id];

            /* Re-project 3D feature and compute error. */
            math::Vec3d x = pose.K * (pose.R * pos3d + pose.t);
            math::Vec2d x2d(x[0] / x[2], x[1] / x[2]);
            total_error += (pos2d - x2d).square_norm();
            num_valid += 1;
        }
        total_error /= static_cast<double>(num_valid);
        all_errors.push_back(std::pair<double, int>(total_error, i));
    }

    if (num_valid_tracks < 2)
        return;

    /* Find the 1/2 percentile. */
    std::size_t const nth_position = all_errors.size() / 2;
    std::nth_element(all_errors.begin(),
        all_errors.begin() + nth_position, all_errors.end());
    double const square_threshold = all_errors[nth_position].first
        * this->opts.track_error_threshold_factor;

    /* Delete all tracks with errors above the threshold. */
    int num_deleted_tracks = 0;
    for (std::size_t i = nth_position; i < all_errors.size(); ++i)
    {
        if (all_errors[i].first > square_threshold)
        {
            this->delete_track(all_errors[i].second);
            num_deleted_tracks += 1;
        }
    }

    if (this->opts.verbose_output)
    {
        float percent = 100.0f * static_cast<float>(num_deleted_tracks)
            / static_cast<float>(num_valid_tracks);
        std::cout << "Deleted " << num_deleted_tracks
            << " of " << num_valid_tracks << " tracks ("
            << util::string::get_fixed(percent, 2)
            << "%) above a threshold of "
            << std::sqrt(square_threshold) << "." << std::endl;
    }
}

/* ---------------------------------------------------------------- */

void
Incremental::normalize_scene (void)
{
    /* Compute AABB for all camera centers. */
    math::Vec3d aabb_min(std::numeric_limits<double>::max());
    math::Vec3d aabb_max(-std::numeric_limits<double>::max());
    math::Vec3d camera_mean(0.0);
    int num_valid_cameras = 0;
    for (std::size_t i = 0; i < this->cameras.size(); ++i)
    {
        CameraPose const& pose = this->cameras[i];
        if (!pose.is_valid())
            continue;
        math::Vec3d center = -(pose.R.transposed() * pose.t);
        for (int j = 0; j < 3; ++j)
        {
            aabb_min[j] = std::min(center[j], aabb_min[j]);
            aabb_max[j] = std::max(center[j], aabb_max[j]);
        }
        camera_mean += center;
        num_valid_cameras += 1;
    }

    /* Compute scale and translation. */
    double scale = 10.0 / (aabb_max - aabb_min).maximum();
    math::Vec3d trans = -(camera_mean / static_cast<double>(num_valid_cameras));

    /* Transform every point. */
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        if (!this->tracks->at(i).is_valid())
            continue;

        this->tracks->at(i).pos = (this->tracks->at(i).pos + trans) * scale;
    }

    /* Transform every camera. */
    for (std::size_t i = 0; i < this->cameras.size(); ++i)
    {
        CameraPose& pose = this->cameras[i];
        if (!pose.is_valid())
            continue;
        pose.t = pose.t * scale - pose.R * trans * scale;
    }
}

/* ---------------------------------------------------------------- */

std::vector<CameraPose> const&
Incremental::get_cameras (void) const
{
    return this->cameras;
}

/* ---------------------------------------------------------------- */

mve::Bundle::Ptr
Incremental::create_bundle (void) const
{
    /* Create bundle data structure. */
    mve::Bundle::Ptr bundle = mve::Bundle::create();
    {
        /* Populate the cameras in the bundle. */
        mve::Bundle::Cameras& bundle_cams = bundle->get_cameras();
        bundle_cams.resize(this->cameras.size());
        for (std::size_t i = 0; i < this->cameras.size(); ++i)
        {
            mve::CameraInfo& cam = bundle_cams[i];
            CameraPose const& pose = this->cameras[i];
            Viewport const& viewport = this->viewports->at(i);
            if (!pose.is_valid())
            {
                cam.flen = 0.0f;
                continue;
            }

            float width = static_cast<float>(viewport.width);
            float height = static_cast<float>(viewport.height);
            float maxdim = std::max(width, height);
            cam.flen = static_cast<float>(pose.get_focal_length());
            cam.flen /= maxdim;
            cam.ppoint[0] = static_cast<float>(pose.K[2]) / width;
            cam.ppoint[1] = static_cast<float>(pose.K[5]) / height;
            std::copy(pose.R.begin(), pose.R.end(), cam.rot);
            std::copy(pose.t.begin(), pose.t.end(), cam.trans);
            cam.dist[0] = viewport.radial_distortion
                * MATH_POW2(pose.get_focal_length());
            cam.dist[1] = 0.0f;
        }

        /* Populate the features in the Bundle. */
        mve::Bundle::Features& bundle_feats = bundle->get_features();
        bundle_feats.reserve(this->tracks->size());
        for (std::size_t i = 0; i < this->tracks->size(); ++i)
        {
            Track const& track = this->tracks->at(i);
            if (!track.is_valid())
                continue;

            /* Copy position and color of the track. */
            bundle_feats.push_back(mve::Bundle::Feature3D());
            mve::Bundle::Feature3D& f3d = bundle_feats.back();
            std::copy(track.pos.begin(), track.pos.end(), f3d.pos);
            f3d.color[0] = track.color[0] / 255.0f;
            f3d.color[1] = track.color[1] / 255.0f;
            f3d.color[2] = track.color[2] / 255.0f;
            f3d.refs.reserve(track.features.size());
            for (std::size_t j = 0; j < track.features.size(); ++j)
            {
                /* For each reference copy view ID, feature ID and 2D pos. */
                f3d.refs.push_back(mve::Bundle::Feature2D());
                mve::Bundle::Feature2D& f2d = f3d.refs.back();
                f2d.view_id = track.features[j].view_id;
                f2d.feature_id = track.features[j].feature_id;

                FeatureSet const& features
                    = this->viewports->at(f2d.view_id).features;
                math::Vec2f const& f2d_pos
                    = features.positions[f2d.feature_id];
                std::copy(f2d_pos.begin(), f2d_pos.end(), f2d.pos);
            }
        }
    }

    return bundle;
}

/* ---------------------------------------------------------------- */

// TODO: Better invalidate the track?
void
Incremental::delete_track (int track_id)
{
    Track& track = this->tracks->at(track_id);
    track.invalidate();

    FeatureReferenceList& ref = track.features;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        int view_id = ref[i].view_id;
        int feature_id = ref[i].feature_id;
        this->viewports->at(view_id).track_ids[feature_id] = -1;
    }
    ref.clear();
}

SFM_BUNDLER_NAMESPACE_END
SFM_NAMESPACE_END
