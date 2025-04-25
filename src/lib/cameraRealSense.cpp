#include "cameraRealSense.hpp"

CameraRealSense::CameraRealSense(Config config) : align(RS2_STREAM_COLOR){

    this->config = config;

    if (config.realtime) {
        this->realtime();
    }
    else {
        this->playback();
    }

    auto profile = pipeline.start(cfg);

    std::cout << "Realsense pipeline started" << std::endl;

    auto color_profile = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    this->width = color_profile.width();
    this->height = color_profile.height();

    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    rs2_intrinsics intrinsics = depth_stream.get_intrinsics();
    K = (cv::Mat_<double>(3,3) << intrinsics.fx,             0, intrinsics.ppx,
                                              0, intrinsics.fy, intrinsics.ppy,
                                              0,             0,              1
        );

    this->accel_threshold = config.accel_threshold;
    this->gyro_threshold = config.gyro_threshold;
}

void CameraRealSense::realtime() {
    std::cout << "Using RealSense device for real-time capture" << std::endl;
    this->cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
    this->cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    this->cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    this->cfg.enable_stream(RS2_STREAM_GYRO,  RS2_FORMAT_MOTION_XYZ32F);
}
    
void CameraRealSense::playback() {
    std::cout << "Using RealSense device for playback" << std::endl;
    this->cfg.enable_device_from_file(config.rosbag_path, false);
}

std::tuple<cv::Mat, cv::Mat, bool> CameraRealSense::get_frames(){

    auto frames = pipeline.wait_for_frames();
    rs2::frameset aligned_frames = align.process(frames);

    rs2::video_frame color_frame = aligned_frames.get_color_frame();
    rs2::depth_frame depth_frame = aligned_frames.get_depth_frame();

    cv::Mat color(cv::Size(this->width, this->height), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
    cv::Mat depth(cv::Size(this->width, this->height), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

    bool moving = this->check_motion(frames);

    return {color, depth, moving};
}

cv::Mat CameraRealSense::create_mask(cv::Mat color, cv::Mat depth){
    /* MASK CREATION FOR KEYPOINTS DETECTION */

    int height = color.rows;
    int width = color.cols;
    
    cv::Mat tri_mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

    cv::Point pt1(width / 2, height * 2 / 3 - 10);
    cv::Point pt2(0 + 20, height);
    cv::Point pt3(width - 20, height);

    std::vector<cv::Point> triangle_cnt = {pt1, pt2, pt3};
    cv::drawContours(tri_mask, std::vector<std::vector<cv::Point>>{triangle_cnt}, 0, cv::Scalar(255), cv::FILLED);
    cv::bitwise_not(tri_mask, tri_mask);

    return tri_mask;
}

bool CameraRealSense::check_motion(const rs2::frameset& frames) {
    bool accel = true;
    bool gyro = true;

    for (const rs2::frame& f : frames) {
        if (auto mf = f.as<rs2::motion_frame>()) {
            rs2_vector data = mf.get_motion_data();
            float magnitude = std::sqrt(data.x*data.x + data.y*data.y + data.z*data.z);
            
            if (mf.get_profile().stream_type() == RS2_STREAM_GYRO) {
                accel = (std::fabs(magnitude) > this->gyro_threshold); //true if moving

                // if (!accel){
                //     std::cout << "Gyro data received with magnitude " << magnitude << std::endl;
                // }
            }
            if (mf.get_profile().stream_type() == RS2_STREAM_ACCEL) {
                gyro = (std::fabs(magnitude - 9.8f) > this->accel_threshold); //true if moving

                // if(!gyro){
                //     std::cout << "Accel data received with magnitude " << magnitude << std::endl;
                // }
            }
        }
    }

    return (accel || gyro);
}

void CameraRealSense::manage_gyroscope(rs2_vector gyro_data) {
    std::cout << "gyroscope message received" << std::endl;
}
void CameraRealSense::manage_accelerometer(rs2_vector accel_data) {
    std::cout << "accelerometer message received" << std::endl;
}

CameraRealSense::~CameraRealSense(){
    pipeline.stop();
    std::cout << "Realsense pipeline stopped" << std::endl;
}
