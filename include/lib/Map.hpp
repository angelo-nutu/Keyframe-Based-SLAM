#pragma once

#include "KeyFrame.hpp"
#include "MapPoint.hpp"

class Map{
public:
    Map(cv::Mat);

    void AddKeyframe(KeyFrame);
    void CreateMapPoints(std::vector<cv::DMatch>);
    
    std::shared_ptr<KeyFrame> GetLastKeyFrame();
    std::vector<cv::Point3d> GetKeyFrames() const;
    std::vector<cv::Point3d> GetMapPoints() const;
    bool IsTrackingEmpty();
private:
    std::vector<std::shared_ptr<KeyFrame>> vecKeyFrames;
    std::shared_ptr<KeyFrame> kfLast;

    cv::Mat K;
    std::vector<cv::Point3d> vecMapPoints;

    void CullKeyframes();
    void UpdateMap();
};