#ifndef VO_HPP
#define VO_HPP

#include <iostream>

#include <librealsense2/rs.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <config.hpp>

class VO {
public:
    VO(Config config);

    void run();
    void extract(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    void match(cv::Mat descriptors1, cv::Mat descriptors2, std::vector<cv::DMatch>& matches);
    void output(cv::Mat color, cv::Mat depth);

    ~VO();

private:
    rs2::pipeline pipeline;
    rs2::config cfg;

    Config config;

    cv::Ptr<cv::Feature2D> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    cv::Mat mask;
};

#endif
