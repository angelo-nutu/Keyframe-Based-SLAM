#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <fstream>
#include <thread>
#include <rerun.hpp>

#include "Utils.hpp"
#include "KeyFrame.hpp"
#include "Map.hpp"
#include "MapPoint.hpp"

class VisualOdometry{
public:
    VisualOdometry(std::pair<cv::Mat, cv::Mat>, std::shared_ptr<Map>);
    // ~VisualOdometry();

    bool Track(cv::Mat, cv::Mat, cv::Mat);
    std::vector<cv::Point3d> GetTrajectory();
private:
    
    cv::Ptr<cv::ORB> ptrExtractor;
    cv::Ptr<cv::BFMatcher> ptrMatcher;
    
    cv::Mat K;
    cv::Mat DistCoeffs;
    
    std::vector<cv::Mat> poses;

    std::shared_ptr<Map> map;

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> ExtractFeatures(cv::Mat, cv::Mat);
    std::pair<std::vector<cv::Point3f>, std::vector<cv::Point2f>> MatchFeatures(cv::Mat, std::vector<cv::KeyPoint>, std::vector<cv::DMatch>&);
    std::tuple<bool, cv::Mat, float> EstimatePose(std::vector<cv::Point3f>, std::vector<cv::Point2f>);
    bool ShouldAddKeyFrame(float);
    void CullKeyframes();
};