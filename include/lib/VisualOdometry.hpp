#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "Utils.hpp"

class VisualOdometry{
public:
    VisualOdometry();
    
private:
    cv::Ptr<cv::xfeatures2d::SURF> ptrExtractor;
    cv::Ptr<cv::FlannBasedMatcher> ptrMatcher;

};