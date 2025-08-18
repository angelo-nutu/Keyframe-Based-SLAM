#pragma once

#include <rerun.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>

class Viewer{
public:
    Viewer();

    void Update(std::vector<Eigen::Vector3d> trajectory, std::vector<Eigen::Vector3d> keyframes, std::vector<Eigen::Vector3d> mapPoints, cv::Mat rgb, cv::Mat depth, cv::Mat mask);
private:
    rerun::RecordingStream recStream;
};