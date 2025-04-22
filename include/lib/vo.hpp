#ifndef VO_HPP
#define VO_HPP

#include <condition_variable>
#include <iostream>
#include <thread>

#include <librealsense2/rs.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "config.hpp"
#include "communication.hpp"
#include "data.h"
#include "plot.hpp"
//#include "mutex.hpp"

class VO {
public:
    VO(Config config);
    // VO(Config config, TelemetryData* tlmData, Communication* communication);

    bool compute(cv::Mat color, cv::Mat depth, cv::Mat mask);
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> feature_extraction(cv::Mat color_gray, std::vector<cv::Mat> mask);
    std::vector<cv::DMatch> feature_matching(cv::Mat descriptors_prev, cv::Mat descriptors, std::vector<cv::KeyPoint> keypoints_prev, std::vector<cv::KeyPoint> keypoints);
    void output(cv::Mat color, cv::Mat depth, cv::Mat match);
    void set_K(cv::Mat K);
    void reset();
    bool compute_pose(std::vector<cv::DMatch> valid_matches, std::vector<cv::KeyPoint> keypoints_prev, std::vector<cv::KeyPoint> keypoints, cv::Mat depth, cv::Mat K);

    ~VO();
    
    struct ExtractionOutput {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    //std::vector<std::future<ExtractionOutput>> VO::feature_extraction(cv::Mat color_gray, std::vector<cv::Mat>& mask);

    std::vector<cv::Mat> poses;
    std::vector<cv::Point2f> trajectory;

    cv::Mat color_gray_prev;
    std::vector<cv::KeyPoint> keypoints_prev;
    cv::Mat descriptors_prev;

private:

    Config config;

    cv::Mat K;

    cv::Ptr<cv::Feature2D> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    bool start;
};

#endif
