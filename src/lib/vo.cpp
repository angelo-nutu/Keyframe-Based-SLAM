#include "vo.hpp"

VO::VO(Config config) {
    config = config;

    if (config.extraction == "orb"){
        extractor = cv::ORB::create();
    } else if (config.extraction == "sift") {
        extractor = cv::SIFT::create();
    } else if (config.extraction == "surf") {
        cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
        surf->setHessianThreshold(400);
        surf->setUpright(true);
        surf->setExtended(true);

        extractor = surf;
    } else {
        std::cerr << "Invalid extraction method provided in the config file." << std::endl;
        exit(1);
    }

    if (config.matching == "bf") {
        matcher = cv::BFMatcher::create();
    } else if (config.matching == "flann") {
        matcher = cv::FlannBasedMatcher::create();
    } else {
        std::cerr << "Invalid matching method provided in the config file." << std::endl;
        exit(1);
    }

    rs2::config cfg;

    if (config.realtime){
        std::cout << "Using RealSense device for real-time capture" << std::endl;

        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    } else {
        std::cout << "Using RealSense device for playback" << std::endl;

        cfg.enable_device_from_file(config.rosbag_path, false);
    }
    
    pipeline.start(cfg);
    std::cout << "VO initialized and Realsense pipeline started" << std::endl;

    rs2::frameset frames = pipeline.wait_for_frames();

    rs2::frame color_frame = frames.get_color_frame();

    const int width = color_frame.as<rs2::video_frame>().get_width();
    const int height = color_frame.as<rs2::video_frame>().get_height();
    mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

    cv::Point pt1(width / 2, height * 2 / 3 - 10);
    cv::Point pt2(0 + 20, height);
    cv::Point pt3(width - 20, height);

    std::vector<cv::Point> triangle_cnt = {pt1, pt2, pt3};

    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{triangle_cnt}, 0, cv::Scalar(255), cv::FILLED);
    cv::fillConvexPoly(mask, triangle_cnt, cv::Scalar(255));

    cv::bitwise_not(mask, mask);
}

void VO::run() {
    rs2::align align(RS2_STREAM_COLOR);

    while (true) {
        rs2::frameset frames = pipeline.wait_for_frames();
        frames = align.process(frames);

        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();

        cv::Mat color(cv::Size(color_frame.as<rs2::video_frame>().get_width(), color_frame.as<rs2::video_frame>().get_height()), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth(cv::Size(depth_frame.as<rs2::video_frame>().get_width(), depth_frame.as<rs2::video_frame>().get_height()), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        cv::Mat color_gray;
        cv::cvtColor(color, color_gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        auto start = std::chrono::steady_clock::now();
        extract(color_gray, keypoints, descriptors);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Extraction took " << duration << " ms" << std::endl;

        cv::drawKeypoints(color, keypoints, color, cv::Scalar::all(-1));

        output(color, depth);
    }
}

void VO::extract(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){
    extractor->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

void VO::output(cv::Mat color, cv::Mat depth) {
    cv::Mat color_rgb; 
    cv::cvtColor(color, color_rgb, cv::COLOR_BGR2RGB);

    cv::Mat depth_u8;
    cv::convertScaleAbs(depth, depth_u8, 0.02);
    cv::applyColorMap(depth_u8, depth_u8, cv::COLORMAP_JET);

    cv::Mat output;
    cv::hconcat(color_rgb, depth_u8, output);
    cv::imshow("Output", output);

    cv::waitKey(1);
}

VO::~VO() {
    pipeline.stop();
    cv::destroyAllWindows();
    std::cout << "Realsense pipeline stopped" << std::endl;
}
