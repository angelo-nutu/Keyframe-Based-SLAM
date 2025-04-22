#include <vo.hpp>
#include <config.hpp>
#include "cameraRealSense.hpp"
#include "mutex.hpp"

#include <chrono>

std::mutex shared_mutex;

int main(int argc, char** argv) {
    
    std::string path;
    
    if (argc > 1) {
        path = argv[1];
    } else {
        std::cerr << "No path provided for the configuration file." << std::endl;
        return 1;
    }

    cv::setNumThreads(0);

    Config config(path);
    VO vo(config);
    CameraRealSense camera(config);
    vo.create_mask(camera.height, camera.width);
    vo.set_K(camera.K);

    bool keep_analyze_frames;
    bool send_data;

    /* TELEMETRY CONFIGURATION */
    TelemetryData* tlmData;
    Communication* communication;
    Plot plt;
    if (config.telemetry) {
        tlmData = new TelemetryData();
        tlmData->create_rotoTranMatrix = false;
        tlmData->start = false;
        tlmData->rotoTranMat = cv::Mat();

        communication = new Communication(config.host, config.vehicleId, tlmData);
        while(communication->getConnection()->getStatus() != PAHOMQTTConnectionStatus::CONNECTED){
            sleep(1);
        }
        std::cout << "Telemetry enabled" << std::endl;

        tlmData->start = true;

        std::cout << "Waiting for a connection with telemetry" << std::endl;

        while (tlmData->rotoTranMat.empty()) {
            sleep(0.1);
        }
        keep_analyze_frames = true;
    } else {
        tlmData = nullptr;
        communication = nullptr;
        std::cout << "Telemetry disabled" << std::endl;

        plt = Plot();
        InitWindow(plt.screenWidth, plt.screenHeight, "Real-Time Trajectory");
        SetTargetFPS(plt.fps);
        keep_analyze_frames = plt.check_condition();
    }

    std::cout << "Main loop started" << std::endl;
    std::cout <<  std::endl << "*************************************************" << std::endl << std::endl;
    
    while(keep_analyze_frames){
        auto [color, depth, moving] = camera.get_frames();
        if (moving){
            shared_mutex.lock();
            if (config.telemetry && tlmData->reset_VO){
                vo.reset();
                tlmData->reset_VO = false;
                std::cout << "VO reset !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            }
            shared_mutex.unlock();
            send_data = vo.compute(color, depth);

            /* DRAW TRAJECTORY */
            if (send_data){
                if (config.telemetry){      /* SEND DATA TO TELEMETRY */
                    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
                    pose.at<double>(0, 3) = vo.poses.back().at<double>(0,3);
                    pose.at<double>(1, 3) = 0;
                    pose.at<double>(2, 3) = -vo.poses.back().at<double>(2,3);

                    shared_mutex.lock();
                    cv::Mat res = tlmData->rotoTranMat * pose;
                    shared_mutex.unlock();
                    communication->sendCoordinates(res.at<double>(0,3), res.at<double>(1,3));                    

                } else {                    /* DRAW THE NEW CAR POSITION WITH RAYLIB */
                    plt.add_point(vo.trajectory.back());
                    plt.draw_plot();
                }
            }

            if (config.debug){
                std::cout <<  std::endl << "*************************************************" << std::endl << std::endl;
            }
        }
    
        if (!config.telemetry){
            keep_analyze_frames = plt.check_condition();
        }
    }

    return 0;
}