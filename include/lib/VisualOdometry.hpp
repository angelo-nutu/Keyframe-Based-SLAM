#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/core/eigen.hpp>

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

    bool Track(cv::Mat, cv::Mat, cv::Mat, bool&);
    std::vector<Eigen::Vector3d> GetTrajectory();
private:
    
    cv::Ptr<cv::ORB> ptrExtractor;
    cv::Ptr<cv::BFMatcher> ptrMatcher;
    
    cv::Mat K;
    cv::Mat DistCoeffs;
    
    std::vector<Sophus::SE3d> poses;

    std::shared_ptr<Map> map;

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> ExtractFeatures(cv::Mat, cv::Mat);
    std::pair<std::vector<cv::Point3d>, std::vector<cv::Point2d>> MatchFeatures(cv::Mat, std::vector<cv::KeyPoint>, std::vector<cv::DMatch>&);
    std::tuple<bool, Sophus::SE3d, float> EstimatePose(std::vector<cv::Point3d>, std::vector<cv::Point2d>);
    bool ShouldAddKeyFrame(float);
};