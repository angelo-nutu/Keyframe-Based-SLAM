#include <Optimizers.hpp>

#include <array>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <memory>
#include <cmath>
#include <iostream>
#include <thread>

using BA = Optimizers::BundleAdjustment;

double BA::ReprojectionErrorCostFunction::fx = 0.0;
double BA::ReprojectionErrorCostFunction::fy = 0.0;
double BA::ReprojectionErrorCostFunction::cx = 0.0;
double BA::ReprojectionErrorCostFunction::cy = 0.0;

BA::ReprojectionErrorCostFunction::ReprojectionErrorCostFunction(double observed_u, double observed_v)
    : u(observed_u), v(observed_v) {}

void BA::ReprojectionErrorCostFunction::SetIntrinsics(double fx_, double fy_, double cx_, double cy_) {
    fx = fx_;
    fy = fy_;
    cx = cx_;
    cy = cy_;
}

BA::DepthPriorCost::DepthPriorCost(double z0_, double w_) : z0(z0_), w(w_) {}

BA::BundleAdjustment::BundleAdjustment(cv::Mat intrinsics, std::shared_ptr<Map> map)
    : map(std::move(map)), numKeyFrames(5) {

    const double fx = intrinsics.at<double>(0, 0);
    const double fy = intrinsics.at<double>(1, 1);
    const double cx = intrinsics.at<double>(0, 2);
    const double cy = intrinsics.at<double>(1, 2);

    ReprojectionErrorCostFunction::SetIntrinsics(fx, fy, cx, cy);
}

void BA::Optimize() {
    if (!this->map || this->map->IsTrackingEmpty()) {
        LOG("The map is currently empty or with insufficient data, skipping optimization.");
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    ceres::Problem problem;
    std::unordered_map<std::shared_ptr<KeyFrame>, double*> kfPoses;        
    std::unordered_map<std::shared_ptr<MapPoint>, double*> mpPoints;
    std::vector<std::shared_ptr<KeyFrame>> keyframes;
    std::unordered_map<std::shared_ptr<MapPoint>, std::pair<int,int>> mpCounts;

    {
        std::lock_guard<std::mutex> lock(gMapMutex);

        keyframes = this->map->GetNKeyFrames(this->numKeyFrames);
        if (keyframes.empty()) {
            LOG("No keyframes available, skipping optimization.");
            return;
        }

    }

    std::vector<std::array<double, 6>> pose_params; pose_params.reserve(keyframes.size());
    std::vector<std::array<double, 3>> point_params; point_params.reserve(1024);


    std::shared_ptr<MapPoint> anchor_mp = nullptr;
    double anchor_z0 = 0.0;

    // size_t k = 0;
    for (auto& kf : keyframes) {
        if (!kf) continue;

        pose_params.emplace_back(); 
        auto& storage = pose_params.back();

        Eigen::Matrix<double,6,1> cam_tangent = kf->sophPose.log();
        for (int j = 0; j < 6; ++j) storage[j] = cam_tangent[j];

        double* cam = storage.data();
        problem.AddParameterBlock(cam, 6);

        // if (k == 0) {
        //     problem.SetParameterBlockConstant(cam);
        // } else {
            // }
        problem.SetManifold(cam, new ceres::EuclideanManifold<6>());

        kfPoses[kf] = cam;
        // ++k;
    }
    problem.SetParameterBlockConstant(kfPoses[keyframes.back()]);

    const double chi2_2d_95 = 5.991;
    const double pixel_sigma = 1.0; 
    const double gate2 = chi2_2d_95 * pixel_sigma * pixel_sigma;

    size_t kf_idx = 0;
    for (auto& kf : keyframes) {
        if (!kf) { ++kf_idx; continue; }

        double* cam = kfPoses[kf];

        for (size_t j = 0; j < kf->vecMapPoints.size(); ++j) {
            auto& mp = kf->vecMapPoints[j];
            if (!mp) continue;

            double* pt = nullptr;
            auto it = mpPoints.find(mp);
            if (it == mpPoints.end()) {
                point_params.emplace_back();
                auto& pstore = point_params.back();
                Eigen::Vector3d Pw = mp->GetPosition();
                pstore[0] = Pw.x(); pstore[1] = Pw.y(); pstore[2] = Pw.z();
                pt = pstore.data();
                problem.AddParameterBlock(pt, 3);
                mpPoints.emplace(mp, pt);
            } else {
                pt = it->second;
            }

            Eigen::Matrix<double,6,1> twc_vec;
            for (int t = 0; t < 6; ++t) twc_vec[t] = cam[t];
            Sophus::SE3d Twc = Sophus::SE3d::exp(twc_vec);

            auto &cnt = mpCounts[mp];
            cnt.first++;

            const Eigen::Vector3d Pw = mp->GetPosition();
            const Eigen::Vector3d Pcam = Twc.inverse() * Pw;
            if (Pcam.z() <= 0.0) {
                LOG("Skip obs: behind camera");
                continue;
            }

            const double u_proj = BA::ReprojectionErrorCostFunction::fx * (Pcam.x() / Pcam.z()) + BA::ReprojectionErrorCostFunction::cx;
            const double v_proj = BA::ReprojectionErrorCostFunction::fy * (Pcam.y() / Pcam.z()) + BA::ReprojectionErrorCostFunction::cy;

            const double u_obs = kf->vecKeypoints[j].pt.x;
            const double v_obs = kf->vecKeypoints[j].pt.y;

            const double du = u_proj - u_obs;
            const double dv = v_proj - v_obs;
            const double err2 = du * du + dv * dv;

            if (err2 > gate2) {
                LOG("Skipped residual by chi2 gate, err2=" << err2 << ")");
                continue;
            }

            cnt.second++;

            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<
                    ReprojectionErrorCostFunction, 2, 6, 3>(
                        new ReprojectionErrorCostFunction(u_obs, v_obs));

            const double alpha = 1.0;
            const double beta  = 0.25;
            const double z     = Pcam.z();
            double w = 1.0 / (alpha + beta * z * z); 

            ceres::LossFunction* huber_loss  = new ceres::HuberLoss(huber_delta);
            ceres::LossFunction* scaled_loss  =
                new ceres::ScaledLoss(huber_loss, w, ceres::TAKE_OWNERSHIP);

            problem.AddResidualBlock(cost, scaled_loss, cam, pt);

            if (kf_idx == 0 && !anchor_mp) {
                anchor_mp = mp;
                anchor_z0 = Pcam.z();
            }
        }
        ++kf_idx;
    }

    if (anchor_mp) {
        double* cam0 = kfPoses[keyframes.front()];
        double* pt_anchor = mpPoints[anchor_mp];

        ceres::CostFunction* depth_prior =
            new ceres::AutoDiffCostFunction<DepthPriorCost, 1, 6, 3>(
                new DepthPriorCost(anchor_z0, 1.0 / (depth_prior_sigma * depth_prior_sigma)));

        problem.AddResidualBlock(depth_prior, nullptr, cam0, pt_anchor);
    } else {
        LOG("No suitable anchor MP found for depth prior; running BA without scale anchor.");
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Setting up the problem takes "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    start = std::chrono::high_resolution_clock::now();

    ceres::Solver::Options options;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.use_inner_iterations = true;
    options.max_num_iterations = 20;
    options.minimizer_progress_to_stdout = false;

    options.num_threads = std::max(1u, std::thread::hardware_concurrency() / 2);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    end = std::chrono::high_resolution_clock::now();
    std::cout << "Solving the problem takes "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    {
        std::lock_guard<std::mutex> lock(gMapMutex);

        const int    min_inlier_kfs   = 2;
        const double min_inlier_ratio = 0.4;

        for (auto& [mp, counts] : mpCounts) {
            if (!mp) continue;
            const int total   = counts.first;
            const int inliers = counts.second;

            bool drop = (total < 2) ||
                        (inliers < min_inlier_kfs) ||
                        (total > 0 && (double)inliers / (double)total < min_inlier_ratio);

            if (!drop) continue;

            for (auto& kf : keyframes) {
                if (!kf) continue;
                for (size_t j = 0; j < kf->vecMapPoints.size(); ++j) {
                    if (kf->vecMapPoints[j] == mp) {
                        kf->vecMapPoints[j].reset();
                    }
                }
            }
        }

        for (auto& [kf, param] : kfPoses) {
            Eigen::Matrix<double,6,1> tangent;
            for (int i = 0; i < 6; ++i) tangent[i] = param[i];
            Sophus::SE3d Twc = Sophus::SE3d::exp(tangent);
            kf->sophPose = Twc;
        }
    
        for (auto& [mp, pt] : mpPoints) {
            Eigen::Vector3d pos(pt[0], pt[1], pt[2]);
            mp->SetPosition(pos);
        }


    }

}
