#include "VisualOdometry.hpp"

VisualOdometry::VisualOdometry(){
    ptrExtractor = cv::xfeatures2d::SURF::create(
        //parameters
    );

    ptrMatcher  = cv::FlannBasedMatcher::create(

    );
}