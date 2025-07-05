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

    void Capture(std::string protocol, std::string port, std::string topicRGBD, std::string topicIntrinsics);

    std::optional<std::tuple<cv::Mat, cv::Mat, cv::Mat>> GrabFrames();
    cv::Mat getK();

private:
    zmq::context_t zmqContext;
    zmq::socket_t  zmqSubscriber;

    cv::Mat matRgbLast;
    cv::Mat matDepthLast;
    cv::Mat matMaskLast;
    bool bAll;

    Intrinsics itrK;
    std::atomic<bool> bKReady;

    std::thread thrCapture;
    std::atomic<bool> bRunThread;
    std::condition_variable cvImgs;
    std::mutex mtxImgs;
};