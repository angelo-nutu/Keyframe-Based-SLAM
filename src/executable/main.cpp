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
    VisualOdometry vo(camera.getK());
    INFO("Initialized Pipeline");

    while (true){
        auto frames = camera.GrabFrames();
        if(!frames){
            INFO("Frames weren't acquired");
            continue;
        }

        cv::Mat output;
        auto [rgb, depth, mask] = *frames;

        vo.Track(rgb, depth, mask);
        
        // cv::convertScaleAbs(depth, depth, 0.02);
        // cv::applyColorMap(depth, depth, cv::COLORMAP_JET);

        // if (mask.channels() == 1) {
        //     cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
        // }

        // cv::hconcat(rgb, depth, output);
        // cv::hconcat(output, mask, output);
        // cv::imshow("Output", output);
        // cv::waitKey(1);
    }
    

    return 0;
}