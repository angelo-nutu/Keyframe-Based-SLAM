#include "VisualOdometry.hpp"

VisualOdometry::VisualOdometry(std::pair<cv::Mat, cv::Mat> intrinsics, std::shared_ptr<Map> map) :
    K(intrinsics.first),
    DistCoeffs(intrinsics.second),
    map(map),
    poses{[] {
            return cv::Mat::eye(4, 4, CV_64F); 
        }()} {
    ptrExtractor = cv::ORB::create(
        2000,                       
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

std::pair<std::vector<cv::Point3f>, std::vector<cv::Point2f>> VisualOdometry::MatchFeatures(cv::Mat dpCurrImg, std::vector<cv::KeyPoint> kpCurrImg, std::vector<cv::DMatch>& matches) {
    std::shared_ptr<KeyFrame> keyframe = this->map->GetLastKeyFrame();

    std::vector<std::vector<cv::DMatch>> knnMatches;
    this->ptrMatcher->knnMatch(keyframe->matDescriptors, dpCurrImg, knnMatches, 2);

    std::vector<MapPoint*> candidates;
    
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> points2D;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < 0.6f * knnMatches[i][1].distance) {
            cv::DMatch match = knnMatches[i][0];
            const cv::KeyPoint& kpPrev = keyframe->vecKeypoints[match.queryIdx];
            const cv::KeyPoint& kpCurr = kpCurrImg[match.trainIdx];

            float z = keyframe->matDepth.at<uint16_t>(cvRound(kpPrev.pt.y), cvRound(kpPrev.pt.x)) * 0.001;
            float y = (kpPrev.pt.y - this->K.at<double>(1, 2)) * z / K.at<double>(1, 1);
            float x = (kpPrev.pt.x - this->K.at<double>(0, 2)) * z / K.at<double>(0, 0);
        
            points3D.push_back(cv::Point3f(x, y, z));
            points2D.push_back(kpCurr.pt);
        }
    }

    return {points3D, points2D};
}

std::tuple<bool, cv::Mat, float> VisualOdometry::EstimatePose(std::vector<cv::Point3f> points3D, std::vector<cv::Point2f> points2D) {
    
    bool success = false;
    cv::Mat rvec, tvec, T;
    std::vector<int> inliers = {0};
    float ratio = 0.0;
    
    success = cv::solvePnPRansac(points3D, points2D, this->K, this->DistCoeffs,
                                    rvec, tvec, false, 100, 8.0, 0.99, inliers);
    
    if (success) {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
    
        cv::Mat tRel = cv::Mat::eye(4, 4, CV_64F);
        R.copyTo(tRel(cv::Rect(0, 0, 3, 3))); 
        tvec.copyTo(tRel(cv::Rect(3, 0, 1, 3))); 
    
        T = this->map->GetLastKeyFrame()->matPose * tRel;
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

    cv::Mat tRel = this->map->GetLastKeyFrame()->matPose.inv() * this->poses.back();

    double dx = tRel.at<double>(0, 3);
    double dy = tRel.at<double>(1, 3);
    double dz = tRel.at<double>(2, 3);
    double translation = std::sqrt(dx*dx + dy*dy + dz*dz);

    addKf |= translation > 0.5;

    cv::Mat rRel = tRel(cv::Rect(0, 0, 3, 3));
    cv::Mat rvec;
    cv::Rodrigues(rRel, rvec);
    double angleRad = cv::norm(rvec);
    double angleDeg = angleRad * 180.0 / CV_PI;

    addKf |= angleDeg > 5;

    addKf |= inliers < 0.4;

    if (addKf)
        framesSinceLastKf = 0;

    return addKf;

}

bool VisualOdometry::Track(cv::Mat rgbFrame, cv::Mat depthFrame, cv::Mat maskFrame){
    if(rgbFrame.empty() || depthFrame.empty()){
        ERROR("The provided images were empty!");
        FIX("Check if the acquisition thread runs succesfully.");
        return false;
    }

    bool success = false, addKeyframe = false;

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

        // create MapPoints based on matches ffs, which actually needs dufddsds 
        // I need matches to create MapPoint basically, since I'll create them solely if I have 2+ observations
        // otherwise it's quite useless. Now to think on how to distinguish a MapPoint with previous observations or not.
        // this should be handled by map lowkey
        // this->map->CreateMapPoints()

    }
    

    return success;
}

std::vector<cv::Point3d> VisualOdometry::GetTrajectory() {
    std::vector<cv::Point3d> trajectory;

    for (auto &&pose : this->poses) {
        trajectory.push_back({pose.at<double>(0, 3), -pose.at<double>(2, 3), -pose.at<double>(1, 3)});
    }
    
    return trajectory;
}
