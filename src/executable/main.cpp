#include <string>
#include <iostream>

#include "VisualOdometry.hpp"
#include "Optimizers.hpp"
#include "Camera.hpp"
#include "Viewer.hpp"
#include "Utils.hpp"
#include "Map.hpp"

int main(int argc, char* argv[]) {
    
    if(argc > 1){
        std::string path = argv[1];
    } else {
        ERROR("No configuration path provided!");
        FIX("Use ./executable path/to/yaml");
        // exit(EXIT_FAILURE); ao lo reimplemento 
    }

    Camera camera;
    std::shared_ptr<Map> map = std::make_shared<Map>(camera.getIntrinsics().first);
    VisualOdometry vo(camera.getIntrinsics(), map);
    Viewer viewer;
    Optimizers::BundleAdjustment localBA(camera.getIntrinsics().first, map);
    INFO("Initialized Pipeline");

    while (true){
        auto frames = camera.GrabFrames();
        if (!frames) {
            INFO("Frames weren't acquired");
            continue;
        }

        auto [rgb, depth, mask] = *frames;

        bool addedKeyFrame;
        bool success = vo.Track(rgb.clone(), depth.clone(), mask.clone(), addedKeyFrame);
        if(!success){
            INFO("A new pose wasn't calculated");
            continue;
        }

        // auto start = std::chrono::high_resolution_clock::now();
        if(addedKeyFrame)
            localBA.Optimize();
        // auto end = std::chrono::high_resolution_clock::now();
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

        viewer.Update(map->GetKeyFramesPositions(), map->GetKeyFramesPositions(), map->GetMapPointsPositions(), rgb.clone(), depth.clone(), mask.clone());

    }
    

    return 0;
}