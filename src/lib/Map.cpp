#include "Map.hpp"

Map::Map(cv::Mat K):
    K(K) {
}

void Map::AddKeyframe(KeyFrame keyFrame){
    static int idCounter = 0;
    keyFrame.id = idCounter++;

    this->kfLast = std::make_shared<KeyFrame>(keyFrame);
    this->vecKeyFrames.push_back(this->kfLast);

    this->CullKeyframes();
    this->UpdateMap();

    std::cout << "Number of MapPoints: " << this->kfLast->vecMapPoints.size() << std::endl;
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

            cv::Mat ptCam = (cv::Mat_<double>(4, 1) << x, y, z, 1.0);
            cv::Mat ptWorld = prevKf->matPose * ptCam;

            cv::Point3d position(
                ptWorld.at<double>(0),           // X
                -ptWorld.at<double>(2),          // -Z
                -ptWorld.at<double>(1)            // Y
            );

            std::shared_ptr<MapPoint> newMapPoint = std::make_shared<MapPoint>(
                position
            );

            LOG("New MapPoint created"); 

            prevKf->vecMapPoints[match.queryIdx] = newMapPoint;
            currKf->vecMapPoints[match.trainIdx] = newMapPoint;
            newMapPoint->AddObservation(prevKf, match.queryIdx);
            newMapPoint->AddObservation(currKf, match.trainIdx);

        } else {
            LOG("New MapPoint created");
            
            currKf->vecMapPoints[match.trainIdx] = prevKf->vecMapPoints[match.queryIdx];
            prevKf->vecMapPoints[match.queryIdx]->AddObservation(currKf, match.trainIdx);
        }
        
    }
}

std::shared_ptr<KeyFrame> Map::GetLastKeyFrame(){
    return this->kfLast;
}

std::vector<cv::Point3d> Map::GetKeyFrames() const{
    std::vector<cv::Point3d> keyframes;

    for (auto &&kf : this->vecKeyFrames) {
        keyframes.push_back({kf->matPose.at<double>(0, 3), -kf->matPose.at<double>(2, 3), -kf->matPose.at<double>(1, 3)});
    }
    
    return keyframes;
}

std::vector<cv::Point3d> Map::GetMapPoints() const{
    return this->vecMapPoints;
}

bool Map::IsTrackingEmpty(){
    return this->vecKeyFrames.empty();
}

void Map::CullKeyframes() {
    if (this->vecKeyFrames.size() <= 5) return;

    std::vector<bool> keep(this->vecKeyFrames.size(), false);
    keep[0] = true;  
    keep.back() = true;  

    for (size_t i = 1; i < this->vecKeyFrames.size()-1; i++) {
        cv::Mat tPrev = this->vecKeyFrames[i-1]->matPose.inv() * this->vecKeyFrames[i]->matPose;
        cv::Mat tNext = this->vecKeyFrames[i]->matPose.inv() * this->vecKeyFrames[i+1]->matPose;

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
        for (size_t i = this->vecKeyFrames.size()-2; 
             i > 0 && keptCount < 5; 
             i--) {
            if (!keep[i]) {
                keep[i] = true;
                keptCount++;
            }
        }
    }

    std::vector<std::shared_ptr<KeyFrame>> new_keyframes;
    for (size_t i = 0; i < this->vecKeyFrames.size(); i++) {
        if (keep[i]) {
            new_keyframes.push_back(this->vecKeyFrames[i]);
        } else {
            std::shared_ptr<KeyFrame> kf = this->vecKeyFrames[i];
            for (auto& mp: kf->vecMapPoints){
                if (mp) {
                    mp->RemoveObservation(kf);

                    if(mp->IsBad()){
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
                LOG("MapPoint Position: " << mp->GetPosition());
                this->vecMapPoints.push_back(mp->GetPosition());
            }
        }
    }
}

