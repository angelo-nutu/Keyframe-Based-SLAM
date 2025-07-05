#include "VisualOdometry.hpp"

VisualOdometry::VisualOdometry(cv::Mat K) :
    K(K) {
    ptrExtractor = cv::xfeatures2d::SURF::create(
        //parameters
    );

    ptrMatcher  = cv::FlannBasedMatcher::create(

    );
}

void VisualOdometry::Track(cv::Mat rgbFrame, cv::Mat depthFrame, cv::Mat maskFrame){
    if(rgbFrame.empty() || depthFrame.empty()){
        ERROR("The provided images were empty!");
        FIX("Check if the acquisition thread runs succesfully.");
        return;
    }

    std::vector<cv::KeyPoint> kpCurrImg;
    cv::Mat                   dpCurrImg;
    cv::cvtColor(rgbFrame, rgbFrame, cv::COLOR_RGB2BGR);

    this->ptrExtractor->detectAndCompute(rgbFrame, maskFrame, kpCurrImg, dpCurrImg);
    if(dpCurrImg.type() != CV_32F){
        dpCurrImg.convertTo(dpCurrImg, CV_32F);
    }

    if(!this->matPrevRgb.empty() && !this->dpPrevImg.empty()){
        std::vector<std::vector<cv::DMatch>> knnMatches;
        this->ptrMatcher->knnMatch(this->dpPrevImg, dpCurrImg, knnMatches, 2);

        
        std::vector<cv::DMatch> matches;
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i][0].distance < 0.7f * knnMatches[i][1].distance) {
                matches.push_back(knnMatches[i][0]);
            }
        }

        std::vector<cv::Point3f> points3D;
        std::vector<cv::Point2f> points2D;

        for (const auto& match : matches) {
            const cv::KeyPoint& kpPrev = this->kpPrevImg[match.queryIdx];
            const cv::KeyPoint& kpCurr = kpCurrImg[match.trainIdx];

            float z = this->matPrevDepth.at<float>(cvRound(kpPrev.pt.y), cvRound(kpPrev.pt.x));
            float y = (kpPrev.pt.y - this->K.at<double>(1, 2)) * z / K.at<double>(1, 1);
            float x = (kpPrev.pt.x - this->K.at<double>(0, 2)) * z / K.at<double>(0, 0);

            points3D.push_back(cv::Point3f(x, y, z));
            points2D.push_back(kpCurr.pt);
        }

        if(points3D.size() >= 4){
            cv::Mat rvec, tvec;
            std::vector<int> inliers;

            bool success = cv::solvePnPRansac(points3D, points2D, K, cv::noArray(),
                                            rvec, tvec, false, 100, 8.0, 0.99, inliers);

            if (success) {
                cv::Mat R;
                cv::Rodrigues(rvec, R);

                std::cout << "Rotation:\n" << R << "\n";
                std::cout << "Translation:\n" << tvec << "\n";
            }
        }

        #ifdef DEBUG
            cv::Mat imgMatches;
            cv::drawMatches(this->matPrevRgb, this->kpPrevImg, rgbFrame, kpCurrImg, matches, imgMatches,
                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            cv::imshow("Matches", imgMatches);
            cv::waitKey(1); 
        #endif
    }

    this->matPrevRgb   = rgbFrame.clone();
    this->matPrevDepth = depthFrame.clone();
    this->kpPrevImg    = kpCurrImg;
    this->dpPrevImg    = dpCurrImg;
}