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
    rs2::pipeline_profile profile = pipeline.start(cfg);

    std::cout << "VO initialized and Realsense pipeline started" << std::endl;

    rs2::frameset frames = pipeline.wait_for_frames();
    rs2::frame color_frame = frames.get_color_frame();

    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    rs2_intrinsics intrinsics = depth_stream.get_intrinsics();

    K = (cv::Mat_<double>(3,3) << intrinsics.fx,  0, intrinsics.ppx,
                                          0, intrinsics.fy, intrinsics.ppy,
                                          0,  0,  1);

    // // Print intrinsics
    // std::cout << "Width: " << intrinsics.width << ", Height: " << intrinsics.height << std::endl;
    // std::cout << "Fx: " << intrinsics.fx << ", Fy: " << intrinsics.fy << std::endl;
    // std::cout << "Cx: " << intrinsics.ppx << ", Cy: " << intrinsics.ppy << std::endl;
    // std::cout << "Distortion Model: " << intrinsics.model << std::endl;
    // std::cout << "Distortion Coefficients: ";
    // for (double coeff : intrinsics.coeffs) {
    //     std::cout << coeff << " ";
    // }
    // std::cout << std::endl;

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

    poses.push_back(cv::Mat::eye(4, 4, CV_64F));
}

void VO::run() {
    rs2::align align(RS2_STREAM_COLOR);

    std::vector<cv::KeyPoint> keypoints_prev;
    cv::Mat descriptors_prev;
    cv::Mat color_gray_prev;

    bool finished = false;
    bool start_vo = true;

    
    const int screenWidth = 800;
    const int screenHeight = 600;
    float minX = std::numeric_limits<float>::infinity(), maxX = - std::numeric_limits<float>::infinity();
    float minY = std::numeric_limits<float>::infinity(), maxY = -std::numeric_limits<float>::infinity();
    float margin = 50.0f, padding = 50.0f;
    float gridSpacingX = 5.0, gridSpacingY = 5.0;

    const float plottingAreaWidth = screenWidth - 2 * margin - 2 * padding;
    const float plottingAreaHeight = screenHeight - 2 * margin - 2 * padding;

    InitWindow(screenWidth, screenHeight, "Real-Time Trajectory");
    SetTargetFPS(60);
    
    while (!WindowShouldClose()) { //true) {
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
        std::cout << "> Feature extraction took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms" << std::endl;

        std::cout << "Number of keypoints: " << keypoints.size() << std::endl;

        if (start_vo) {
            finished = true;
            start_vo = false;
        } else {
            /* FEATURE MATCHING */
            cv::setNumThreads(-1);
            std::vector<cv::DMatch> matches;
            auto overhead_start = std::chrono::high_resolution_clock::now();
            matcher->match(descriptors_prev, descriptors, matches);
            auto overhead_end = std::chrono::high_resolution_clock::now();
            std::cout << "> Feature matching took "
                        << std::chrono::duration_cast<std::chrono::milliseconds>(overhead_end - overhead_start).count()
                        << " ms" << std::endl;

            std::cout << "Number of matches found: " << matches.size() << std::endl;

            std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
                return a.distance < b.distance;
            });

            std::vector<cv::DMatch> valid_matches;
            // std::vector<cv::Point2f> src_pts;
            // std::vector<cv::Point2f> dst_pts;
            for (const auto& m : matches) {
                if (m.queryIdx >= 0 && m.queryIdx < static_cast<int>(keypoints_prev.size()) &&
                    m.trainIdx >= 0 && m.trainIdx < static_cast<int>(keypoints.size())) {
                    
                    // src_pts.push_back(keypoints_prev[m.queryIdx].pt);
                    // dst_pts.push_back(keypoints[m.trainIdx].pt);
                    valid_matches.push_back(m);
                }
            }
            
            std::sort(valid_matches.begin(), valid_matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
                return a.distance < b.distance; 
            });

            std::cout << "Number of valid matches: " << valid_matches.size() << std::endl;

            /* VISUALIZE KEYPOINTS and MATCHING */
            cv::drawKeypoints(color, keypoints, color, cv::Scalar::all(-1));
            cv::Mat img_matches;
            cv::drawMatches(color_gray_prev, keypoints_prev, color_gray, keypoints, valid_matches, img_matches);
            output(color, depth, img_matches);

            /* PNP */

            if (valid_matches.size() > 0 && keypoints_prev.size() > 0 && keypoints.size() > 0) {

                std::vector<cv::Point3f> obj_points;
                std::vector<cv::Point2f> img_points;
                
                for (cv::DMatch match : valid_matches) {
                    if (match.queryIdx < keypoints_prev.size() &&  match.trainIdx < keypoints.size() ) {
                        cv::KeyPoint keypoint = keypoints_prev[match.queryIdx];
                        int u = static_cast<int>(keypoint.pt.x); 
                        int v = static_cast<int>(keypoint.pt.y);

                        if (u > 0 && v > 0 && u < depth_frame.as<rs2::video_frame>().get_height() && v < depth_frame.as<rs2::video_frame>().get_width()) {
                            double depth_value = depth.at<uint16_t>(v, u) * 0.001;;
                            
                            if (depth_value > 0) {
                                double x = (u - K.at<double>(0,2)) * depth_value / K.at<double>(0,0);
                                double y = (v - K.at<double>(1,2)) * depth_value / K.at<double>(1,1);
                                obj_points.push_back(cv::Point3f(x, y, depth_value));
                                img_points.push_back(cv::Point2f(keypoints[match.trainIdx].pt));

                            }
                        }
                    }
                }

                if (obj_points.size() >= 10) {
                    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);  // Assuming no distortion
                    // TODO: decide whether to keep the previous line here, or to take it out of the while-loop
                    cv::Mat rvec, tvec, R, T;
                    std::vector<int> inliers;
                    auto overhead_start = std::chrono::high_resolution_clock::now();
                    bool success = cv::solvePnPRansac(obj_points, img_points, K, distCoeffs, rvec, tvec, inliers);
                    auto overhead_end = std::chrono::high_resolution_clock::now();
                    std::cout << "> PnP took "
                                << std::chrono::duration_cast<std::chrono::milliseconds>(overhead_end - overhead_start).count()
                                << " ms" << std::endl;
                    
                    std::cout << "Number of inliers found: " << inliers.size() << std::endl;

                    if (success ) {
                        cv::Rodrigues(rvec, R);
                        T = tvec;
                        finished = true;

                        /* COMPUTE THE POSE */
                        cv::Mat T_rel = cv::Mat::eye(4, 4, CV_64F);
                        R.copyTo(T_rel(cv::Rect(0, 0, 3, 3))); 
                        T.copyTo(T_rel(cv::Rect(3, 0, 1, 3))); 

                        cv::Mat T = poses.back() * T_rel;
                        poses.push_back(T);
                        cv::Point2f last = cv::Point2f(T.at<double>(0,3), -T.at<double>(2,3));
                        trajectory.push_back(last);

                        /* DRAW THE NEW CAR POSITION */

                        minX = (last.x < minX) ? last.x : minX;
                        maxX = (last.x > maxX) ? last.x : maxX;
                        minY = (last.y < minY) ? last.y : minY;
                        maxY = (last.y > maxY) ? last.y : maxY;

                        float scaleX = (screenWidth - 2 * margin - 2 * padding) / (maxX - minX);
                        float scaleY = (screenHeight - 2 * margin - 2 * padding) / (maxY - minY);
                        float scale = (scaleX < scaleY) ? scaleX : scaleY;
                        
                        BeginDrawing();
                        ClearBackground(BLACK);

                        // draw x grid
                        float converted_margin_X = (- padding - (scale == scaleY ? (plottingAreaWidth - maxX + minX)/2 : 0.0))/scale + minX;
                        float startX = std::ceil(converted_margin_X / gridSpacingX) * gridSpacingX;
                        float endX = (screenWidth - margin - margin - padding - (scale == scaleY ? (plottingAreaWidth - maxX + minX)/2 : 0.0))/scale + minX;
                        for (float i = startX; i <= endX; i += gridSpacingX) {
                            float linePlot = margin + padding + (i - minX) * scale + (scale == scaleY ? (plottingAreaWidth - maxX + minX)/2 : 0.0);
                            DrawLine(linePlot, margin, linePlot, screenHeight - margin, DARKGRAY);
                
                            char xLabel[16];
                            sprintf(xLabel, "%.1fm", i);
                            DrawText(xLabel, linePlot - 15, screenHeight - margin + 5, 12, DARKGRAY);
                        }
                        
                        // draw z grid
                        float converted_margin_Y = (screenHeight - (screenHeight - margin) - margin - padding - (scale == scaleX ? (plottingAreaHeight - maxY + minY)/2 : 0.0))/scale + minY;                      
                        float startY = std::ceil(converted_margin_Y / gridSpacingY) * gridSpacingY;
                        float endY = (screenHeight - (margin) - margin - padding - (scale == scaleX ? (plottingAreaHeight - maxY + minY)/2 : 0.0))/scale + minY;
                        for (float i = startY; i <= endY; i += gridSpacingY) {
                            float linePlot = screenHeight - (margin + padding + (i - minY) * scale + (scale == scaleX ? (plottingAreaHeight - maxY + minY)/2 : 0.0));
                            DrawLine(margin, linePlot, screenWidth - margin, linePlot, DARKGRAY);

                            char yLabel[16];
                            sprintf(yLabel, "%.1fm", i);
                            DrawText(yLabel, margin - 45, linePlot - 5, 12, DARKGRAY);
                        }

                        // draw axis
                        DrawLine(margin, screenHeight - margin, screenWidth - margin, screenHeight - margin, WHITE);
                        DrawLine(margin, margin, margin, screenHeight - margin, WHITE); 
                        DrawText("x (meters)", screenWidth - 100, screenHeight - margin + 10, 14, WHITE);
                        DrawText("z (meters)", margin - 40, margin - 20, 14, WHITE);

                        // draw points
                        for (cv::Point2f point : trajectory) {
                            float screenX = margin + padding + (point.x - minX) * scale + (scale == scaleY ? (plottingAreaWidth - maxX + minX)/2 : 0);
                            float screenY = screenHeight - (margin + padding + (point.y - minY) * scale + (scale == scaleX ? (plottingAreaHeight - maxY + minY)/2 : 0));
                            DrawCircle((int)screenX, (int)screenY, 3, YELLOW);
                        }
                        float screenX = margin + padding + (last.x - minX) * scale + (scale == scaleY ? (plottingAreaWidth - maxX + minX)/2 : 0);
                        float screenY = screenHeight - (margin + padding + (last.y - minY) * scale + (scale == scaleX ? (plottingAreaHeight - maxY + minY)/2 : 0));
                        DrawCircle((int)screenX, (int)screenY, 4, BLUE);
                
                        EndDrawing();
                    }

                }
            }
        }

        if (finished) {
            color_gray_prev = color_gray.clone();
            keypoints_prev = keypoints;
            descriptors_prev = descriptors;
            finished = false;
        }

        std::cout <<  std::endl << "*************************************************" << std::endl << std::endl;

    }
}


void VO::output(cv::Mat color, cv::Mat depth, cv::Mat match) {
    cv::Mat color_rgb;
    cv::cvtColor(color, color_rgb, cv::COLOR_BGR2RGB);

    cv::Mat depth_u8;
    cv::convertScaleAbs(depth, depth_u8, 0.02);
    cv::applyColorMap(depth_u8, depth_u8, cv::COLORMAP_JET);

    cv::Mat top_row;
    cv::hconcat(color_rgb, depth_u8, top_row);

    cv::Mat output;
    cv::vconcat(top_row, match, output);

    cv::imshow("Output", output);

    cv::waitKey(1);
}

VO::~VO() {
    pipeline.stop();
    cv::destroyAllWindows();
    CloseWindow();

    std::cout << "Realsense pipeline stopped" << std::endl;
}