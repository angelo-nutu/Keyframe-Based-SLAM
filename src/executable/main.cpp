#include <string>
#include <iostream>

#include "VisualOdometry.hpp"
#include "Camera.hpp"
#include "Utils.hpp"

int main(int argc, char* argv[]) {
    
    if(argc > 1){
        std::string path = argv[1];
    } else {
        ERROR("No configuration path provided!");
        FIX("Use ./executable path/to/yaml");
        // exit(EXIT_FAILURE); ao lo reimplemento 
    }

    Camera camera;

    while (true){
        auto frames = camera.GrabFrames();
        if(!frames){
            INFO("Frames weren't acquired");
            continue;
        }

        cv::Mat output;
        auto [rgb, depth] = *frames;

        cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);

        cv::convertScaleAbs(depth, depth, 0.02);
        cv::applyColorMap(depth, depth, cv::COLORMAP_JET);

        cv::hconcat(rgb, depth, output);
        cv::imshow("Output", output);
        cv::waitKey(1);
    }
    

    return 0;
}