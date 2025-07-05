#pragma once

#include <thread>
#include <atomic>
#include <condition_variable>

#include <zmq.hpp>
#include <opencv2/opencv.hpp>

#include "Utils.hpp"
#include "as-serializers/Camera.h"

class Camera{
public:
    Camera();
    ~Camera();

    void Capture(std::string protocol, std::string port, std::string topic);

    std::optional<std::pair<cv::Mat, cv::Mat>> GrabFrames();

private:
    zmq::context_t zmqContext;
    zmq::socket_t  zmqSubscriber;

    cv::Mat matRgbLast;
    cv::Mat matDepthLast;
    bool bBoth;

    std::thread thrCapture;
    std::atomic<bool> bRunThread;
    std::condition_variable cvImgs;
    std::mutex mtxImgs;
};