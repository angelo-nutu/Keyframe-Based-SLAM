#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <fstream>
#include <thread>
#include <rerun.hpp>

#include "Utils.hpp"

class VisualOdometry{
public:
    VisualOdometry(std::pair<cv::Mat, cv::Mat>);
    // ~VisualOdometry();

    std::optional<std::vector<cv::Point2d>> Track(cv::Mat rgbFrame, cv::Mat depthFrame, cv::Mat maskFrame);
    
private:
    cv::Ptr<cv::xfeatures2d::SURF> ptrExtractor;
    cv::Ptr<cv::FlannBasedMatcher> ptrMatcher;

    cv::Mat matPrevRgb, matPrevDepth;
    std::vector<cv::KeyPoint> kpPrevImg;
    cv::Mat dpPrevImg;

    cv::Mat K;
    cv::Mat DistCoeffs;

    std::vector<cv::Mat> poses;
};