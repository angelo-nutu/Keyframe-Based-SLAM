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
#include <raylib.h>

class VO {
public:
    VO(Config config);

    void run();
    
    void output(cv::Mat color, cv::Mat depth, cv::Mat match);
    
    ~VO();
    
    struct ExtractionOutput {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

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

};

#endif
