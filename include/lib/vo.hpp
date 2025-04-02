#ifndef VO_HPP
#define VO_HPP

#include <condition_variable>
#include <iostream>
#include <thread>

#include <librealsense2/rs.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <config.hpp>
//#include <raylib.h>

#include "communication.hpp"
#include "data.h"
#include "plot.hpp"

class VO {
public:
    VO(Config config);
    // VO(Config config, TelemetryData* tlmData, Communication* communication);

    void run();
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> feature_extraction(cv::Mat color_gray, int n);
    std::vector<cv::DMatch> feature_matching(cv::Mat descriptors_prev, cv::Mat descriptors, std::vector<cv::KeyPoint> keypoints_prev, std::vector<cv::KeyPoint> keypoints);
    void output(cv::Mat color, cv::Mat depth, cv::Mat match);
    std::pair<bool, cv::Mat> compute_pose(std::vector<cv::DMatch> valid_matches, std::vector<cv::KeyPoint> keypoints_prev, std::vector<cv::KeyPoint> keypoints, rs2::frame depth_frame, cv::Mat depth);

    ~VO();
    
    struct ExtractionOutput {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    //std::vector<std::future<ExtractionOutput>> VO::feature_extraction(cv::Mat color_gray, std::vector<cv::Mat>& mask);

private:
    rs2::pipeline pipeline;
    rs2::config cfg;

    Config config;

    cv::Ptr<cv::Feature2D> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    std::vector<cv::Mat> mask;
    cv::Mat K;

    std::vector<cv::Mat> poses;
    std::vector<cv::Point2f> trajectory;

    TelemetryData* tlmData;
    Communication* communication;

    Plot plt;
};

#endif
