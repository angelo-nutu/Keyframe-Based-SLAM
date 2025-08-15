#pragma once

#include <opencv2/opencv.hpp>
#include "Utils.hpp"
#include "MapPoint.hpp"

class KeyFrame{
public:
    KeyFrame(){};
    KeyFrame(cv::Mat frame, cv::Mat depth, std::vector<cv::KeyPoint> kps, cv::Mat des, cv::Mat pose):
    matFrame(frame),
    matDepth(depth),
    vecKeypoints(kps),
    vecMapPointObserved(kps.size(), false),
    vecMapPoints(kps.size(), nullptr),
    matDescriptors(des),
    matPose(pose) {};

    KeyFrame(const KeyFrame& other) {
        this->id = other.id;
        this->matFrame = other.matFrame;
        this->matDepth = other.matDepth;
        this->vecKeypoints = other.vecKeypoints;
        this->vecMapPointObserved = other.vecMapPointObserved;
        this->vecMapPoints = other.vecMapPoints;
        this->matDescriptors = other.matDescriptors;
        this->matPose = other.matPose;
    }


    int id;
    cv::Mat matFrame;
    cv::Mat matDepth;
    std::vector<cv::KeyPoint> vecKeypoints;
    std::vector<bool> vecMapPointObserved;
    std::vector<std::shared_ptr<MapPoint>> vecMapPoints;
    cv::Mat matDescriptors;
    cv::Mat matPose;
};