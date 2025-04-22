#ifndef CAMERAREALSENSE_HPP
#define CAMERAREALSENSE_HPP

#include <iostream>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include "cameraManagement.hpp"
#include "config.hpp"

class CameraRealSense : public CameraManagement{
    public:
        CameraRealSense(Config config);
        std::tuple<cv::Mat, cv::Mat, bool> get_frames() override;
        bool check_motion(const rs2::frameset& frames);
        cv::Mat create_mask(cv::Mat color, cv::Mat depth) override;
        ~CameraRealSense();

        int width;
        int height;
        cv::Mat K;

    private:

        void realtime() override;
        void playback() override;

        Config config;

        rs2::pipeline pipeline;
        rs2::config cfg;
        rs2::pipeline_profile profile;
        
        rs2::align align;

        bool moving;
        void manage_gyroscope(rs2_vector gyro_data);
        void manage_accelerometer(rs2_vector accel_data);

        float accel_threshold;
        float gyro_threshold;

};

#endif
