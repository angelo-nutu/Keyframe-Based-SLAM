#pragma once

#include <rerun.hpp>
#include <opencv2/opencv.hpp>

class Viewer{
public:
    Viewer();

    void update(std::vector<cv::Point3d> trajectory, std::vector<cv::Point3d> keyframes, std::vector<cv::Point3d> mapPoints, cv::Mat rgb, cv::Mat depth, cv::Mat mask);
private:
    rerun::RecordingStream recStream;
};