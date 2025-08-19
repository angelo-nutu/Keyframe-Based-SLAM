#include "Map.hpp"

Map::Map(cv::Mat K):
    K(K) {
}

void Map::AddKeyframe(KeyFrame keyFrame){
    static int idCounter = 0;
    keyFrame.id = idCounter++;

    this->kfLast = std::make_shared<KeyFrame>(keyFrame);
    this->vecKeyFrames.push_back(this->kfLast);

}

void Map::CreateMapPoints(std::vector<cv::DMatch> matches) {
    
    if(matches.empty()) return;
    
    std::shared_ptr<KeyFrame> prevKf = this->vecKeyFrames[this->vecKeyFrames.size() - 2];
    std::shared_ptr<KeyFrame> currKf = this->vecKeyFrames[this->vecKeyFrames.size() - 1];


    for (const auto& match: matches) {
        const cv::KeyPoint& kpPrev = prevKf->vecKeypoints[match.queryIdx];
        const cv::KeyPoint& kpCurr = currKf->vecKeypoints[match.trainIdx];

        if (!prevKf->vecMapPoints[match.queryIdx]) {
            float u = kpPrev.pt.x;
            float v = kpPrev.pt.y;
            uint16_t d = prevKf->matDepth.at<uint16_t>(cvRound(v), cvRound(u));

            float z = d * 0.001f;
            float fx = K.at<double>(0, 0);
            float fy = K.at<double>(1, 1);
            float cx = K.at<double>(0, 2);
            float cy = K.at<double>(1, 2);

            float x = (u - cx) * z / fx;
            float y = (v - cy) * z / fy;

            Eigen::Vector3d ptCam(x, y, z);

            Eigen::Vector3d ptWorld = prevKf->sophPose * ptCam;


            std::shared_ptr<MapPoint> newMapPoint = std::make_shared<MapPoint>(
                ptWorld
            );

            prevKf->vecMapPoints[match.queryIdx] = newMapPoint;
            currKf->vecMapPoints[match.trainIdx] = newMapPoint;
            newMapPoint->AddObservation(prevKf, match.queryIdx);
            newMapPoint->AddObservation(currKf, match.trainIdx);

        } else {
            currKf->vecMapPoints[match.trainIdx] = prevKf->vecMapPoints[match.queryIdx];
            prevKf->vecMapPoints[match.queryIdx]->AddObservation(currKf, match.trainIdx);
        }
    }


    // this->CullKeyframes();  
    this->UpdateMap();
}


std::shared_ptr<KeyFrame> Map::GetLastKeyFrame(){
    return this->kfLast;
}

std::vector<std::shared_ptr<KeyFrame>> Map::GetNKeyFrames(int win) const{
    if (this->vecKeyFrames.size() >= win) {
        return std::vector<std::shared_ptr<KeyFrame>>(this->vecKeyFrames.end() - win, this->vecKeyFrames.end());
    }
    
    return this->vecKeyFrames;
}

std::vector<Eigen::Vector3d> Map::GetKeyFramesPositions() const{
    std::vector<Eigen::Vector3d> keyframes;

    for (auto &&kf : this->vecKeyFrames) {
        Eigen::Vector3d t = kf->sophPose.translation();

        keyframes.emplace_back(t.x(), -t.z(), -t.y());
    }
    
    return keyframes;
}

std::vector<Eigen::Vector3d> Map::GetMapPointsPositions() const{
    std::vector<Eigen::Vector3d> mapPoints;

    for (auto &&mp : this->vecMapPoints) {
        Eigen::Vector3d pt(mp.x(), -mp.z(), -mp.y());
        
        mapPoints.emplace_back(pt);
    }
    return mapPoints;
}

bool Map::IsTrackingEmpty(){
    return this->vecKeyFrames.empty();
}

void Map::CullKeyframes() {
    if (this->vecKeyFrames.size() <= 5) return;

    std::vector<bool> keep(this->vecKeyFrames.size(), false);
    keep[0] = true;  
    keep.back() = true;  

    for (size_t i = 1; i < this->vecKeyFrames.size() - 1; ++i) {
        auto& prevPose = this->vecKeyFrames[i - 1]->sophPose;
        auto& currPose = this->vecKeyFrames[i]->sophPose;
        auto& nextPose = this->vecKeyFrames[i + 1]->sophPose;

        Sophus::SE3d tPrev = prevPose.inverse() * currPose;
        Sophus::SE3d tNext = currPose.inverse() * nextPose;

        double trPrev = tPrev.translation().norm();
        double trNext = tNext.translation().norm();

        double rotPrev = tPrev.so3().log().norm();
        double rotNext = tNext.so3().log().norm();

        const double trThresh = 1.5;
        const double rotThresh = 0.1;

        keep[i] = (trPrev > trThresh || trNext > trThresh ||
                   rotPrev > rotThresh || rotNext > rotThresh);
    }

    size_t keptCount = std::count(keep.begin(), keep.end(), true);
    if (keptCount < 5) {
        for (size_t i = this->vecKeyFrames.size() - 2;
             i > 0 && keptCount < 5;
             --i) {
            if (!keep[i]) {
                keep[i] = true;
                ++keptCount;
            }
        }
    }

    std::vector<std::shared_ptr<KeyFrame>> new_keyframes;
    for (size_t i = 0; i < this->vecKeyFrames.size(); ++i) {
        if (keep[i]) {
            new_keyframes.push_back(this->vecKeyFrames[i]);
        } else {
            std::shared_ptr<KeyFrame> kf = this->vecKeyFrames[i];
            for (auto& mp : kf->vecMapPoints) {
                if (mp) {
                    mp->RemoveObservation(kf);

                    if (mp->IsBad()) {
                        auto observations = mp->GetObservations();
                        for (auto& obs : observations) {
                            if (auto kfPtr = obs.first.lock()) {
                                kfPtr->vecMapPoints[obs.second].reset();
                            }
                        }
                    }

                    mp.reset();
                }
            }
        }
    }

    this->vecKeyFrames = new_keyframes;

    if (!this->vecKeyFrames.empty()) {
        kfLast = this->vecKeyFrames.back();
    }
}

void Map::UpdateMap() {

    this->vecMapPoints.clear();

    for (const auto& kf : this->vecKeyFrames) {
        
        for (const auto& mp : kf->vecMapPoints) {
            if (mp) {
                this->vecMapPoints.push_back(mp->GetPosition());
            }
        }
    }
}