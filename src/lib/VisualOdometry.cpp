#include "VisualOdometry.hpp"

VisualOdometry::VisualOdometry(std::pair<cv::Mat, cv::Mat> intrinsics) :
    K(intrinsics.first),
    DistCoeffs(intrinsics.second),
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

std::pair<std::vector<cv::Point3f>, std::vector<cv::Point2f>> VisualOdometry::MatchFeatures(cv::Mat dpCurrImg, std::vector<cv::KeyPoint> kpCurrImg) {
    KeyFrame keyframe = this->kfLast;

    std::vector<std::vector<cv::DMatch>> knnMatches;
    this->ptrMatcher->knnMatch(keyframe.matDescriptors, dpCurrImg, knnMatches, 2);
    
    std::vector<cv::DMatch> matches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < 0.6f * knnMatches[i][1].distance) {
            matches.push_back(knnMatches[i][0]);
        }
    }
    
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> points2D;
    
    for (const auto& match : matches) {
        const cv::KeyPoint& kpPrev = keyframe.vecKeypoints[match.queryIdx];
        const cv::KeyPoint& kpCurr = kpCurrImg[match.trainIdx];
    
        float z = keyframe.matDepth.at<uint16_t>(cvRound(kpPrev.pt.y), cvRound(kpPrev.pt.x)) * 0.001;
        float y = (kpPrev.pt.y - this->K.at<double>(1, 2)) * z / K.at<double>(1, 1);
        float x = (kpPrev.pt.x - this->K.at<double>(0, 2)) * z / K.at<double>(0, 0);
    
        points3D.push_back(cv::Point3f(x, y, z));
        points2D.push_back(kpCurr.pt);
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
    
        T = this->kfLast.matPose * tRel;
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

    cv::Mat tRel = this->kfLast.matPose.inv() * this->poses.back();

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

void VisualOdometry::CullKeyframes() {
    if (vecOdometry.size() <= 5) return;

    std::vector<bool> keep(vecOdometry.size(), false);
    keep[0] = true;  
    keep.back() = true;  

    for (size_t i = 1; i < vecOdometry.size()-1; i++) {
        cv::Mat tPrev = vecOdometry[i-1].matPose.inv() * vecOdometry[i].matPose;
        cv::Mat tNext = vecOdometry[i].matPose.inv() * vecOdometry[i+1].matPose;
        
        double trPrev = cv::norm(tPrev(cv::Rect(3, 0, 1, 3)));
        double trNext = cv::norm(tNext(cv::Rect(3, 0, 1, 3)));
        
        cv::Mat rvecPrev, rvecNext;
        cv::Rodrigues(tPrev(cv::Rect(0, 0, 3, 3)), rvecPrev);
        cv::Rodrigues(tNext(cv::Rect(0, 0, 3, 3)), rvecNext);
        
        double rotPrev = cv::norm(rvecPrev);
        double rotNext = cv::norm(rvecNext);

        const double trTresh = 1.5;  
        const double rotThresh = 0.1; 
        keep[i] = (trPrev > trTresh || 
                       trNext > trTresh ||
                       rotPrev > rotThresh || 
                       rotNext > rotThresh);
    }

    size_t keptCount = std::count(keep.begin(), keep.end(), true);
    if (keptCount < 5) {
        for (size_t i = vecOdometry.size()-2; 
             i > 0 && keptCount < 5; 
             i--) {
            if (!keep[i]) {
                keep[i] = true;
                keptCount++;
            }
        }
    }

    std::vector<KeyFrame> new_keyframes;
    for (size_t i = 0; i < vecOdometry.size(); i++) {
        if (keep[i]) {
            new_keyframes.push_back(vecOdometry[i]);
        }
    }

    vecOdometry = new_keyframes;
    if (!vecOdometry.empty()) {
        kfLast = vecOdometry.back();
    }
}

bool VisualOdometry::Track(cv::Mat rgbFrame, cv::Mat depthFrame, cv::Mat maskFrame){
    if(rgbFrame.empty() || depthFrame.empty()){
        ERROR("The provided images were empty!");
        FIX("Check if the acquisition thread runs succesfully.");
        return false;
    }

    bool success = false, addKeyframe = false;

    auto [kpCurrImg, dpCurrImg] = this->ExtractFeatures(rgbFrame, maskFrame);

    if(!this->vecOdometry.empty()){
        auto [points3D, points2D] = this->MatchFeatures(dpCurrImg, kpCurrImg);
        
        if(points3D.size() >= 4) {
            auto [estimated, T, inliers] = this->EstimatePose(points3D, points2D);
            success = estimated;
            
            addKeyframe = this->ShouldAddKeyFrame(inliers);
            
        }
    }
    
    if (this->vecOdometry.empty() || addKeyframe) {
        this->kfLast = {
            rgbFrame,
            depthFrame,
            kpCurrImg,
            dpCurrImg,
            this->poses.back()
        };
        
        vecOdometry.push_back(this->kfLast);
        this->CullKeyframes();
    }
    

    return success;
}

std::vector<cv::Point2d> VisualOdometry::GetTrajectory() {
    std::vector<cv::Point2d> trajectory;

    for (auto &&pose : this->poses) {
        trajectory.push_back({pose.at<double>(0, 3), -pose.at<double>(2, 3)});
    }
    
    return trajectory;
}

std::vector<cv::Point2d> VisualOdometry::GetKeyFrames() {
    std::vector<cv::Point2d> keyframes;

    for (auto &&kf : this->vecOdometry) {
        keyframes.push_back({kf.matPose.at<double>(0, 3), -kf.matPose.at<double>(2, 3)});
    }
    
    return keyframes;
}
