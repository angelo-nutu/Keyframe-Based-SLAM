#pragma once

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "Utils.hpp"
#include "MapPoint.hpp"

class KeyFrame{
public:
    KeyFrame(){};
    KeyFrame(cv::Mat frame, cv::Mat depth, std::vector<cv::KeyPoint> kps, cv::Mat des, Sophus::SE3d pose):
    matFrame(frame),
    matDepth(depth),
    vecKeypoints(kps),
    vecMapPointObserved(kps.size(), false),
    vecMapPoints(kps.size(), nullptr),
    matDescriptors(des),
    sophPose(pose) {};

    KeyFrame(const KeyFrame& other) {
        this->id = other.id;
        this->matFrame = other.matFrame;
        this->matDepth = other.matDepth;
        this->vecKeypoints = other.vecKeypoints;
        this->vecMapPointObserved = other.vecMapPointObserved;
        this->vecMapPoints = other.vecMapPoints;
        this->matDescriptors = other.matDescriptors;
        this->sophPose = other.sophPose;
    }


    int id;
    cv::Mat matFrame;
    cv::Mat matDepth;
    std::vector<cv::KeyPoint> vecKeypoints;
    std::vector<bool> vecMapPointObserved;
    std::vector<std::shared_ptr<MapPoint>> vecMapPoints;
    cv::Mat matDescriptors;
    Sophus::SE3d sophPose;
};