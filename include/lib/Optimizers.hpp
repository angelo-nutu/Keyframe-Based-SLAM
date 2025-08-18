#pragma once

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include <Eigen/Core>

#include "Map.hpp"

namespace Optimizers {

    class BundleAdjustment {
    public:
        BundleAdjustment(cv::Mat intrinsics, std::shared_ptr<Map> map);
        void Optimize();

    private:
        struct ReprojectionErrorCostFunction {
            ReprojectionErrorCostFunction(double observed_u, double observed_v);

            template <typename T>
            bool operator()(const T* const cam,
                            const T* const pt,
                            T* residual) const {
                Eigen::Matrix<T, 6, 1> twc_vec(cam[0], cam[1], cam[2], cam[3], cam[4], cam[5]);
                Sophus::SE3<T> Twc = Sophus::SE3<T>::exp(twc_vec);

                Eigen::Matrix<T, 3, 1> Pw(pt[0], pt[1], pt[2]);
                Eigen::Matrix<T, 3, 1> Pcam = Twc.inverse() * Pw;

                const T eps = T(1e-8);
                T z = std::max(Pcam[2], eps);

                T u_proj = T(fx) * (Pcam[0] / z) + T(cx);
                T v_proj = T(fy) * (Pcam[1] / z) + T(cy);

                residual[0] = u_proj - T(u);
                residual[1] = v_proj - T(v);
                return true;
            }

            static void SetIntrinsics(double fx_, double fy_, double cx_, double cy_);
            static double fx, fy, cx, cy;

            double u;
            double v;
        };

   
        struct DepthPriorCost {
            DepthPriorCost(double z0_, double w_);
            template <typename T>
            bool operator()(const T* const cam0,   
                            const T* const pt,    
                            T* residual) const {
                Eigen::Matrix<T, 6, 1> twc0_vec(cam0[0], cam0[1], cam0[2], cam0[3], cam0[4], cam0[5]);
                Sophus::SE3<T> Twc0 = Sophus::SE3<T>::exp(twc0_vec);

                Eigen::Matrix<T,3,1> Pw(pt[0], pt[1], pt[2]);
                Eigen::Matrix<T,3,1> Pcam0 = Twc0.inverse() * Pw;

                residual[0] = T(std::sqrt(w)) * (Pcam0[2] - T(z0));
                return true;
            }
            double z0;
            double w;
        };

    private:
        int numKeyFrames = 5;
        std::shared_ptr<Map> map;

        double huber_delta = 1.0;         
        double depth_prior_sigma = 0.30; 
    };

} 
