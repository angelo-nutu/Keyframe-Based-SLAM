#ifndef PLOT_HPP
#define PLOT_HPP

#include <iostream>

#include <opencv2/opencv.hpp>
#include <raylib.h>

class Plot{    
public:
    Plot();
    float World2Pixel(float world_value, bool ace);
    float Pixel2World(float pixel_value, bool ace);
    void add_point(cv::Point2f point);
    void draw_plot();
    bool check_condition();

    ~Plot();

    float gridSpacingX;
    float gridSpacingY;

    int screenWidth;
    int screenHeight;
    int fps;

private:

    float margin;
    float padding;

    float plottingAreaWidth;
    float plottingAreaHeight;

    float minX;
    float maxX;
    float minY;
    float maxY;

    float scaleX;
    float scaleY;
    float scale;

    std::vector<cv::Point2f> trajectory;
};

#endif
