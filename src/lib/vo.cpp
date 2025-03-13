#include "vo.hpp"
#include <future>
#include <atomic>

VO::VO(Config config) {
    cv::setNumThreads(0);

    /* CONFIG FILE, BASIC ODOMETRY SETUP*/
    int n = std::thread::hardware_concurrency();

    config = config;

    if (config.extraction == "orb") {
        extractor = cv::ORB::create();
    } else if (config.extraction == "sift") {
        extractor = cv::SIFT::create(2000);
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
        if (config.extraction == "orb") {
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        }else if (config.extraction == "sift" or config.extraction == "surf") {
            matcher = cv::BFMatcher::create(cv::NORM_L2, true);
        }
    } else if (config.matching == "flann") {
        matcher = cv::FlannBasedMatcher::create();
    } else {
        std::cerr << "Invalid matching method provided in the config file." << std::endl;
        exit(1);
    }

    /* CAMERA CONFIGURATION PARAMETERS */
    rs2::config cfg;

    if (config.realtime) {
        std::cout << "Using RealSense device for real-time capture" << std::endl;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    } else {
        std::cout << "Using RealSense device for playback" << std::endl;
        cfg.enable_device_from_file(config.rosbag_path, false);
    }

    /* MASK CREATION FOR KEYPOINTS DETECTION */
    pipeline.start(cfg);
    std::cout << "VO initialized and Realsense pipeline started" << std::endl;

    rs2::frameset frames = pipeline.wait_for_frames();
    rs2::frame color_frame = frames.get_color_frame();

    const int width = color_frame.as<rs2::video_frame>().get_width();
    const int height = color_frame.as<rs2::video_frame>().get_height();
    cv::Mat tri_mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

    cv::Point pt1(width / 2, height * 2 / 3 - 10);
    cv::Point pt2(0 + 20, height);
    cv::Point pt3(width - 20, height);

    std::vector<cv::Point> triangle_cnt = {pt1, pt2, pt3};
    cv::drawContours(tri_mask, std::vector<std::vector<cv::Point>>{triangle_cnt}, 0, cv::Scalar(255), cv::FILLED);
    cv::bitwise_not(tri_mask, tri_mask);

    for (int i = 0; i < n; ++i) {
        int start_row = MAX(0, i * (height / n) - 0.3 * (height / n));
        int end_row = MIN(height, (i == n - 1) ? height + 0.3 * (height / n) : (i + 1) * (height / n) + 0.3 * (height / n));
        cv::Mat mask_portion = tri_mask(cv::Range(start_row, end_row), cv::Range::all());
        mask.push_back(mask_portion);
    }
}

void VO::run() {
    rs2::align align(RS2_STREAM_COLOR);

    std::vector<cv::KeyPoint> keypoints_prev;
    cv::Mat descriptors_prev;
    cv::Mat color_prev;
    

    while (true) {
        rs2::frameset frames = pipeline.wait_for_frames();
        frames = align.process(frames);

        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();

        cv::Mat color(cv::Size(color_frame.as<rs2::video_frame>().get_width(), color_frame.as<rs2::video_frame>().get_height()), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth(cv::Size(depth_frame.as<rs2::video_frame>().get_width(), depth_frame.as<rs2::video_frame>().get_height()), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        cv::Mat color_gray;
        cv::cvtColor(color, color_gray, cv::COLOR_BGR2GRAY);

        /* PARALLELIZE FEATURE EXTRACTION */

        auto start = std::chrono::high_resolution_clock::now();

        int height = color_gray.rows;
        int n = std::thread::hardware_concurrency();

        std::vector<std::future<ExtractionOutput>> futures;
        for (int i = 0; i < mask.size(); ++i) {
            int start_row = MAX(0, i * (height / n) - 0.3 * (height / n));
            int end_row = MIN(height, (i == n - 1) ? height + 0.3 * (height / n) : (i + 1) * (height / n) + 0.3 * (height / n));
            cv::Mat image_portion = color_gray(cv::Range(start_row, end_row), cv::Range::all()).clone();

            futures.push_back(std::async(std::launch::async, [this, image_portion, i, start_row] {
                std::vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;

                
                auto overhead_start = std::chrono::high_resolution_clock::now();
                extractor->detectAndCompute(image_portion, mask[i], keypoints, descriptors);
                auto overhead_end = std::chrono::high_resolution_clock::now();
                std::cout << "Feature extraction overhead for portion " << i << ": "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(overhead_end - overhead_start).count()
                          << " ms" << std::endl;
                

                for (auto& keypoint : keypoints) {
                    keypoint.pt.y += start_row;
                }

                return ExtractionOutput{keypoints, descriptors};
            }));
        }

        /* COLLECT RESULTS */
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        for (auto& future : futures) {
            ExtractionOutput output = future.get();
            keypoints.insert(keypoints.end(), output.keypoints.begin(), output.keypoints.end());
            descriptors.push_back(output.descriptors);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Feature extraction took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms" << std::endl;

        std::cout << "Number of keypoints: " << keypoints.size() << std::endl;

        /* VISUALIZE KEYPOINTS */
        cv::drawKeypoints(color, keypoints, color, cv::Scalar::all(-1));
        output(color, depth);

        if (!keypoints_prev.empty() && !descriptors_prev.empty()){
            /* FEATURE MATCHING */
            std::vector<cv::DMatch> matches;
            auto overhead_start = std::chrono::high_resolution_clock::now();
            matcher->match(descriptors_prev, descriptors, matches);
            auto overhead_end = std::chrono::high_resolution_clock::now();
            std::cout << "Feature matching overhead: "
                        << std::chrono::duration_cast<std::chrono::milliseconds>(overhead_end - overhead_start).count()
                        << " ms" << std::endl;

            std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
                return a.distance < b.distance;
            });

            std::vector<cv::DMatch> valid_matches;
            std::vector<cv::Point2f> src_pts;
            std::vector<cv::Point2f> dst_pts;
            for (const auto& m : matches) {
                if (m.queryIdx >= 0 && m.queryIdx < static_cast<int>(keypoints_prev.size()) &&
                    m.trainIdx >= 0 && m.trainIdx < static_cast<int>(keypoints.size())) {
                    
                    src_pts.push_back(keypoints_prev[m.queryIdx].pt);
                    dst_pts.push_back(keypoints[m.trainIdx].pt);
                    valid_matches.push_back(m);
                }
            }
            
            std::sort(valid_matches.begin(), valid_matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
                return a.distance < b.distance; 
            });

            cv::Mat img_matches;
            cv::drawMatches(color_prev, keypoints_prev, color, keypoints, valid_matches, img_matches);

            cv::imshow("Matches", img_matches);
            cv::waitKey(1);

        }
        color.copyTo(color_prev);
        keypoints_prev = keypoints;
        descriptors_prev = descriptors;

    }
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