#ifndef CAMERAMANAGEMENT_HPP
#define CAMERAMANAGEMENT_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

class CameraManagement {
public:

    virtual std::tuple<cv::Mat, cv::Mat, bool> get_frames() = 0;
    virtual cv::Mat create_mask(cv::Mat color, cv::Mat depth) = 0;

private:

    virtual void realtime() = 0;
    virtual void playback() = 0;

};

#endif
