#include <string>
#include <iostream>

#include "VisualOdometry.hpp"
#include "Optimizers.hpp"
#include "Camera.hpp"
#include "Viewer.hpp"
#include "Utils.hpp"
#include "Map.hpp"

std::mutex gMapMutex;

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

    std::atomic<bool> baRunning = false;
    std::thread       localBAThread;
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

        if(addedKeyFrame && !baRunning.load()) {
            baRunning.store(true);
            localBAThread = std::thread([&](){
                localBA.Optimize();
                baRunning.store(false);
            });
            localBAThread.detach();
        }
        // if(addedKeyFrame){
        //     localBA.Optimize();
        // }
        

        std::vector<Eigen::Vector3d> KfsPos;
        std::vector<Eigen::Vector3d> MpsPos;

        {   
            std::lock_guard<std::mutex> lock(gMapMutex);
            KfsPos = map->GetKeyFramesPositions(); 
            MpsPos = map->GetMapPointsPositions();
        }
        viewer.Update(KfsPos, KfsPos, MpsPos, rgb.clone(), depth.clone(), mask.clone());

    }
    

    return 0;
}