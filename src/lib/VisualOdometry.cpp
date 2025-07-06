#include "VisualOdometry.hpp"

VisualOdometry::VisualOdometry(std::pair<cv::Mat, cv::Mat> intrinsics) :
    K(intrinsics.first),
    DistCoeffs(intrinsics.second),
    poses{[] {
            return cv::Mat::eye(4, 4, CV_64F); 
        }()} {
    ptrExtractor = cv::xfeatures2d::SURF::create(
        30, 6, 4, true, false
    );

    ptrMatcher  = cv::FlannBasedMatcher::create();
}

std::optional<std::vector<cv::Point2d>> VisualOdometry::Track(cv::Mat rgbFrame, cv::Mat depthFrame, cv::Mat maskFrame){
    if(rgbFrame.empty() || depthFrame.empty()){
        ERROR("The provided images were empty!");
        FIX("Check if the acquisition thread runs succesfully.");
        return std::nullopt;
    }

    bool                     success = false;
    std::vector<cv::Point2d> trajectory;

    std::vector<cv::KeyPoint> kpCurrImg;
    cv::Mat                   dpCurrImg;
    cv::cvtColor(rgbFrame, rgbFrame, cv::COLOR_RGB2BGR);

    this->ptrExtractor->detectAndCompute(rgbFrame, maskFrame, kpCurrImg, dpCurrImg);
    if(dpCurrImg.type() != CV_32F){
        dpCurrImg.convertTo(dpCurrImg, CV_32F);
    }

    if(!this->matPrevRgb.empty() && !this->dpPrevImg.empty()){
        std::vector<std::vector<cv::DMatch>> knnMatches;
        this->ptrMatcher->knnMatch(this->dpPrevImg, dpCurrImg, knnMatches, 2);

        
        std::vector<cv::DMatch> matches;
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i][0].distance < 0.7f * knnMatches[i][1].distance) {
                matches.push_back(knnMatches[i][0]);
            }
        }

        std::vector<cv::Point3f> points3D;
        std::vector<cv::Point2f> points2D;

        for (const auto& match : matches) {
            const cv::KeyPoint& kpPrev = this->kpPrevImg[match.queryIdx];
            const cv::KeyPoint& kpCurr = kpCurrImg[match.trainIdx];

            float z = this->matPrevDepth.at<uint16_t>(cvRound(kpPrev.pt.y), cvRound(kpPrev.pt.x)) * 0.001;
            float y = (kpPrev.pt.y - this->K.at<double>(1, 2)) * z / K.at<double>(1, 1);
            float x = (kpPrev.pt.x - this->K.at<double>(0, 2)) * z / K.at<double>(0, 0);

            points3D.push_back(cv::Point3f(x, y, z));
            points2D.push_back(kpCurr.pt);
        }

        

        if(points3D.size() >= 4){
            cv::Mat rvec, tvec;
            std::vector<int> inliers;

            success = cv::solvePnPRansac(points3D, points2D, this->K, this->DistCoeffs,
                                            rvec, tvec, false, 100, 8.0, 0.99, inliers);

            if (success) {
                cv::Mat R;
                cv::Rodrigues(rvec, R);

                cv::Mat T_rel = cv::Mat::eye(4, 4, CV_64F);
                R.copyTo(T_rel(cv::Rect(0, 0, 3, 3))); 
                tvec.copyTo(T_rel(cv::Rect(3, 0, 1, 3))); 

                cv::Mat T = poses.back() * T_rel;
                this->poses.push_back(T.clone());

                
                for (auto &&T : poses) {
                    trajectory.push_back(cv::Point2f(T.at<double>(0,3), -T.at<double>(2,3)));
                }
                

                // std::ofstream file("trajectory.txt");
                // if (!file.is_open()) {
                //     std::cerr << "Failed to open trajectory.txt for writing." << std::endl;
                //     return;
                // }

                // for (const auto& pt : trajectory) {
                //     file << pt.x << " " << pt.y << "\n";
                // }

                // file.close();
                // std::cout << "Trajectory saved to trajectory.txt (" << trajectory.size() << " points)." << std::endl;
            }
        }

        // std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    
    this->matPrevRgb   = rgbFrame.clone();
    this->matPrevDepth = depthFrame.clone();
    this->kpPrevImg    = kpCurrImg;
    this->dpPrevImg    = dpCurrImg;

    if(success){
        return trajectory;
    }
    return std::nullopt;
}