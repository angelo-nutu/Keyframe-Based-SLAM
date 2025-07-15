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

    bool Track(cv::Mat, cv::Mat, cv::Mat);
    std::vector<cv::Point2d> GetTrajectory();
    std::vector<cv::Point2d> GetKeyFrames();
private:
    struct KeyFrame {
        cv::Mat matFrame;
        cv::Mat matDepth;
        std::vector<cv::KeyPoint> vecKeypoints;
        cv::Mat matDescriptors;
        cv::Mat matPose;
    };

    
    // cv::Ptr<cv::xfeatures2d::SURF> ptrExtractor;
    // cv::Ptr<cv::FlannBasedMatcher> ptrMatcher;
    cv::Ptr<cv::ORB> ptrExtractor;
    cv::Ptr<cv::BFMatcher> ptrMatcher;
    
    // cv::Mat matPrevRgb, matPrevDepth;
    // std::vector<cv::KeyPoint> kpPrevImg;
    // cv::Mat dpPrevImg;
    
    cv::Mat K;
    cv::Mat DistCoeffs;
    
    std::vector<cv::Mat> poses;
    std::vector<VisualOdometry::KeyFrame> vecOdometry;
    VisualOdometry::KeyFrame kfLast;

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> ExtractFeatures(cv::Mat, cv::Mat);
    std::pair<std::vector<cv::Point3f>, std::vector<cv::Point2f>> MatchFeatures(cv::Mat, std::vector<cv::KeyPoint>);
    std::tuple<bool, cv::Mat, float> EstimatePose(std::vector<cv::Point3f>, std::vector<cv::Point2f>);
    bool ShouldAddKeyFrame(float);
    void CullKeyframes();
};