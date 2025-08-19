#pragma once

#include <Eigen/Core>
#include <unordered_set>
#include <algorithm>

#include "KeyFrame.hpp"
#include "MapPoint.hpp"

class Map{
public:
    Map(cv::Mat);

    void AddKeyframe(KeyFrame);
    void CreateMapPoints(std::vector<cv::DMatch>);
    
    std::shared_ptr<KeyFrame> GetLastKeyFrame();
    std::vector<std::shared_ptr<KeyFrame>> GetNKeyFrames(int) const;
    std::vector<Eigen::Vector3d> GetKeyFramesPositions() const;
    std::vector<Eigen::Vector3d> GetMapPointsPositions() const;
    bool IsTrackingEmpty();
    void UpdateMap();

private:
    std::vector<std::shared_ptr<KeyFrame>> vecKeyFrames;
    std::shared_ptr<KeyFrame> kfLast;
    
    cv::Mat K;
    std::vector<Eigen::Vector3d> vecMapPoints;
    void CullKeyframes();

};