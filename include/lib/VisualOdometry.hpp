#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "Utils.hpp"

class VisualOdometry{
public:
    VisualOdometry(cv::Mat K);

    void Track(cv::Mat rgbFrame, cv::Mat depthFrame, cv::Mat maskFrame);
    
private:
    cv::Ptr<cv::xfeatures2d::SURF> ptrExtractor;
    cv::Ptr<cv::FlannBasedMatcher> ptrMatcher;

    cv::Mat matPrevRgb, matPrevDepth;
    std::vector<cv::KeyPoint> kpPrevImg;
    cv::Mat dpPrevImg;

    cv::Mat K;
};