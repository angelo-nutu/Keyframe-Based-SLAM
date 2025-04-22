#include "vo.hpp"
#include <future>
#include <atomic>

VO::VO(Config config){

    /* CONFIG FILE, BASIC ODOMETRY SETUP*/

    this->config = config;

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

    this->start = true;
    poses.push_back(cv::Mat::eye(4, 4, CV_64F));

    std::cout << "VO initialized" << std::endl;
}

bool VO::compute(cv::Mat color, cv::Mat depth) {
    
    cv::Mat color_gray;
    cv::cvtColor(color, color_gray, cv::COLOR_BGR2GRAY);

    /* PARALLEL FEATURE EXTRACTION */
    auto start = std::chrono::high_resolution_clock::now();
    int n = std::thread::hardware_concurrency();
    cv::setNumThreads(0);
    auto [keypoints, descriptors] = this->feature_extraction(color_gray);
    auto end = std::chrono::high_resolution_clock::now();

    if(config.debug){
        std::cout << "> Feature extraction took "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                    << " ms" << std::endl;
        std::cout << "Number of keypoints: " << keypoints.size() << std::endl;
    }

    bool update_prev_variables = true;
    bool success = false;

    if (!this->start) {

        /* FEATURE MATCHING */
        cv::setNumThreads(-1);
        std::vector<cv::DMatch> valid_matches = this->feature_matching(this->descriptors_prev, descriptors, this->keypoints_prev, keypoints);
        
        if (config.debug){
            std::cout << "Number of valid matches: " << valid_matches.size() << std::endl;
        }

        /* VISUALIZE KEYPOINTS and MATCHING */
        cv::drawKeypoints(color, keypoints, color, cv::Scalar::all(-1));
        cv::Mat img_matches;
        cv::drawMatches(color_gray_prev, keypoints_prev, color_gray, keypoints, valid_matches, img_matches);

        if (config.display){
            output(color, depth, img_matches);
        }

        /* COMPUTE POSE */
        success = compute_pose(valid_matches, keypoints_prev, keypoints, depth, this->K);
        update_prev_variables = success;

    } else {
        this->start = false;
    }

    if (update_prev_variables) {
        this->color_gray_prev = color_gray.clone();
        this->keypoints_prev = keypoints;
        this->descriptors_prev = descriptors;
    }

    return success;
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> VO::feature_extraction(cv::Mat color_gray){
    int height = color_gray.rows;
    
    std::vector<std::future<ExtractionOutput>> futures;
    int n = std::thread::hardware_concurrency();
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

            if (config.debug){
                std::cout << "Feature extraction overhead for portion " << i << ": "
                            << std::chrono::duration_cast<std::chrono::milliseconds>(overhead_end - overhead_start).count()
                            << " ms" << std::endl;
            }

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

    return {keypoints, descriptors};
}

std::vector<cv::DMatch> VO::feature_matching(cv::Mat descriptors_prev, cv::Mat descriptors, std::vector<cv::KeyPoint> keypoints_prev, std::vector<cv::KeyPoint> keypoints) {
    std::vector<cv::DMatch> matches;
    auto overhead_start = std::chrono::high_resolution_clock::now();
    matcher->match(descriptors_prev, descriptors, matches);
    auto overhead_end = std::chrono::high_resolution_clock::now();

    if (config.debug){
        std::cout << "> Feature matching took "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(overhead_end - overhead_start).count()
                    << " ms" << std::endl;

        std::cout << "Number of matches found: " << matches.size() << std::endl;
    }

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

    return valid_matches;
}

void VO::create_mask(int height, int width){

    /* MASK CREATION FOR KEYPOINTS DETECTION */
    
    cv::Mat tri_mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

    cv::Point pt1(width / 2, height * 2 / 3 - 10);
    cv::Point pt2(0 + 20, height);
    cv::Point pt3(width - 20, height);

    std::vector<cv::Point> triangle_cnt = {pt1, pt2, pt3};
    cv::drawContours(tri_mask, std::vector<std::vector<cv::Point>>{triangle_cnt}, 0, cv::Scalar(255), cv::FILLED);
    cv::bitwise_not(tri_mask, tri_mask);

    int n = std::thread::hardware_concurrency();
    for (int i = 0; i < n; ++i) {
        int start_row = MAX(0, i * (height / n) - 0.3 * (height / n));
        int end_row = MIN(height, (i == n - 1) ? height + 0.3 * (height / n) : (i + 1) * (height / n) + 0.3 * (height / n));
        cv::Mat mask_portion = tri_mask(cv::Range(start_row, end_row), cv::Range::all());
        this->mask.push_back(mask_portion);
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

void VO::reset(){
    this->start = true;
    this->poses.clear();
    this->poses.push_back(cv::Mat::eye(4, 4, CV_64F));
    this->trajectory.clear();
    this->color_gray_prev.release();
    this->keypoints_prev.clear();
    this->descriptors_prev.release();
}

bool VO::compute_pose(std::vector<cv::DMatch> valid_matches, std::vector<cv::KeyPoint> keypoints_prev, std::vector<cv::KeyPoint> keypoints, cv::Mat depth, cv::Mat K){
    if (valid_matches.size() > 0 && keypoints_prev.size() > 0 && keypoints.size() > 0) {

        std::vector<cv::Point3f> obj_points;
        std::vector<cv::Point2f> img_points;
        
        for (cv::DMatch match : valid_matches) {
            if (match.queryIdx < keypoints_prev.size() &&  match.trainIdx < keypoints.size() ) {
                cv::KeyPoint keypoint = keypoints_prev[match.queryIdx];
                int u = static_cast<int>(keypoint.pt.x); 
                int v = static_cast<int>(keypoint.pt.y);

                if (u > 0 && v > 0 && u < depth.rows && v < depth.cols) {
                    double depth_value = depth.at<uint16_t>(v, u) * 0.001;
                    
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
            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            auto overhead_start = std::chrono::high_resolution_clock::now();
            bool success = cv::solvePnPRansac(obj_points, img_points, K, distCoeffs, rvec, tvec, inliers);
            auto overhead_end = std::chrono::high_resolution_clock::now();

            if (config.debug){
                std::cout << "> PnP took "
                            << std::chrono::duration_cast<std::chrono::milliseconds>(overhead_end - overhead_start).count()
                            << " ms" << std::endl;
                
                std::cout << "Number of inliers found: " << inliers.size() << std::endl;
            }

            if (success) {
                cv::Mat R;
                cv::Rodrigues(rvec, R);

                /* COMPUTE THE POSE */
                cv::Mat T_rel = cv::Mat::eye(4, 4, CV_64F);
                R.copyTo(T_rel(cv::Rect(0, 0, 3, 3))); 
                tvec.copyTo(T_rel(cv::Rect(3, 0, 1, 3))); 

                cv::Mat T = poses.back() * T_rel;
                poses.push_back(T);
                cv::Point2f last = cv::Point2f(T.at<double>(0,3), -T.at<double>(2,3));
                trajectory.push_back(last);

                return true;
            }
        }
    }
    return false;
}

void VO::set_K(cv::Mat K){
    this->K = K;
}

VO::~VO() {
    cv::destroyAllWindows();
}