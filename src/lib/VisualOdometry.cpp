#include "VisualOdometry.hpp"

VisualOdometry::VisualOdometry(std::pair<cv::Mat, cv::Mat> intrinsics, std::shared_ptr<Map> map) :
    K(intrinsics.first),
    DistCoeffs(intrinsics.second),
    map(map),
    poses{[] {
            return Sophus::SE3d(); 
        }()} {
    ptrExtractor = cv::ORB::create(
        3000,                       
        1.2f, 8, 31, 0, 2,          
        cv::ORB::HARRIS_SCORE,      
        31,                         
        20                          
    );

    ptrMatcher = cv::BFMatcher::create(cv::NORM_HAMMING);
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> VisualOdometry::ExtractFeatures(cv::Mat rgb, cv::Mat mask) {

    std::vector<cv::KeyPoint> kpCurrImg;
    cv::Mat                   dpCurrImg;
    cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);
    
    this->ptrExtractor->detectAndCompute(rgb, mask, kpCurrImg, dpCurrImg);

    return {kpCurrImg, dpCurrImg};
}

std::pair<std::vector<cv::Point3d>, std::vector<cv::Point2d>> VisualOdometry::MatchFeatures(cv::Mat dpCurrImg, std::vector<cv::KeyPoint> kpCurrImg, std::vector<cv::DMatch>& matches) {
    std::shared_ptr<KeyFrame> keyframe = this->map->GetLastKeyFrame();

    std::vector<std::vector<cv::DMatch>> knnMatches;
    this->ptrMatcher->knnMatch(keyframe->matDescriptors, dpCurrImg, knnMatches, 2);

    std::vector<MapPoint*> candidates;
    
    std::vector<cv::Point3d> points3D;
    std::vector<cv::Point2d> points2D;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < 0.6f * knnMatches[i][1].distance) {
            cv::DMatch match = knnMatches[i][0];
            matches.push_back(match);
            const cv::KeyPoint& kpPrev = keyframe->vecKeypoints[match.queryIdx];
            const cv::KeyPoint& kpCurr = kpCurrImg[match.trainIdx];

            float z = keyframe->matDepth.at<uint16_t>(cvRound(kpPrev.pt.y), cvRound(kpPrev.pt.x)) * 0.001;
            float y = (kpPrev.pt.y - this->K.at<double>(1, 2)) * z / K.at<double>(1, 1);
            float x = (kpPrev.pt.x - this->K.at<double>(0, 2)) * z / K.at<double>(0, 0);
        
            points3D.push_back(cv::Point3d(x, y, z));
            points2D.push_back(kpCurr.pt);
        }
    }

    return {points3D, points2D};
}

std::tuple<bool, Sophus::SE3d, float> VisualOdometry::EstimatePose(std::vector<cv::Point3d> points3D, std::vector<cv::Point2d> points2D) {
    
    bool success = false;
    cv::Mat rvec, tvec;
    Sophus::SE3d T;
    std::vector<int> inliers = {0};
    float ratio = 0.0;
    
    success = cv::solvePnPRansac(points3D, points2D, this->K, this->DistCoeffs,
                                    rvec, tvec, false, 100, 8.0, 0.99, inliers);
    
    if (success) {
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);

        Eigen::Matrix3d R;
        cv::cv2eigen(R_cv, R);

        Eigen::Vector3d t;
        cv::cv2eigen(tvec, t);

        Sophus::SE3d T_cw = Sophus::SE3d(R, t);
        Sophus::SE3d T_wc = T_cw.inverse(); 

        T = this->map->GetLastKeyFrame()->sophPose * T_wc;
        this->poses.push_back(T);

        ratio = static_cast<float>(inliers.size()) / points2D.size();
    
    }

    return {success, T, ratio};

}

bool VisualOdometry::ShouldAddKeyFrame(float inliers){
    bool addKf = false;

    static int framesSinceLastKf = 0;
    framesSinceLastKf++;

    if (framesSinceLastKf < 10)
        return false;

    Sophus::SE3d tRel = this->map->GetLastKeyFrame()->sophPose.inverse() * this->poses.back();

    Eigen::Vector3d trans = tRel.translation();
    double translation = trans.norm();

    addKf |= translation > 0.5;

    double angleRad = tRel.so3().log().norm();
    double angleDeg = angleRad * 180.0 / M_PI;

    addKf |= angleDeg > 5;

    addKf |= inliers < 0.4;

    if (addKf)
        framesSinceLastKf = 0;

    return addKf;
}


bool VisualOdometry::Track(cv::Mat rgbFrame, cv::Mat depthFrame, cv::Mat maskFrame, bool &addKeyframe){
    if(rgbFrame.empty() || depthFrame.empty()){
        ERROR("The provided images were empty!");
        FIX("Check if the acquisition thread runs succesfully.");
        return false;
    }

    bool success = false;

    auto [kpCurrImg, dpCurrImg] = this->ExtractFeatures(rgbFrame, maskFrame);

    std::vector<cv::DMatch> matches;
    if(!this->map->IsTrackingEmpty()){
        auto [points3D, points2D] = this->MatchFeatures(dpCurrImg, kpCurrImg, matches);
        
        if(points3D.size() >= 4) {
            auto [estimated, T, inliers] = this->EstimatePose(points3D, points2D);
            success = estimated;
            
            addKeyframe = this->ShouldAddKeyFrame(inliers);
            
        }
    }
    
    if (this->map->IsTrackingEmpty() || addKeyframe) {
        KeyFrame keyFrame(
            rgbFrame,
            depthFrame,
            kpCurrImg,
            dpCurrImg,
            this->poses.back()
        );

        this->map->AddKeyframe(keyFrame);
        this->map->CreateMapPoints(matches);
        
    }
    

    return success;
}

std::vector<Eigen::Vector3d> VisualOdometry::GetTrajectory() {
    std::vector<Eigen::Vector3d> trajectory;
    trajectory.reserve(this->poses.size());

    for (const auto& pose : this->poses) {
        Eigen::Vector3d t = pose.translation();

        Eigen::Vector3d transformed_t(t.x(), -t.z(), -t.y());

        trajectory.push_back(transformed_t);
    }

    return trajectory;
}

