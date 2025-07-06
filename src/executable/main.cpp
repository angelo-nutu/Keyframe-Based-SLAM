#include <string>
#include <iostream>

#include "VisualOdometry.hpp"
#include "Camera.hpp"
#include "Viewer.hpp"
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
    VisualOdometry vo(camera.getIntrinsics());
    Viewer viewer;
    INFO("Initialized Pipeline");

    while (true){
        auto frames = camera.GrabFrames();
        if (!frames) {
            INFO("Frames weren't acquired");
            continue;
        }

        auto [rgb, depth, mask] = *frames;

        auto trajectory = vo.Track(rgb.clone(), depth.clone(), mask.clone());
        if(!trajectory){
            INFO("A new pose wasn't calculated");
            continue;
        }

        viewer.update(*trajectory, rgb.clone(), depth.clone(), mask.clone());


    }
    

    return 0;
}