#include <vo.hpp>
#include <config.hpp>
#include <raylib.h>

TelemetryData tlmData;

int main(int argc, char** argv) {
    
    std::string path;
    
    if (argc > 1) {
        path = argv[1];
    } else {
        std::cerr << "No path provided for the configuration file." << std::endl;
        return 1;
    }

    tlmData.create_rotoTranMatrix = false;
    tlmData.start = false;

    Config config(path);

    if (config.telemetry){
        std::string host = "localhost";
        std::string vehicleId = "giorgia";
        Communication communication(host, vehicleId, &tlmData);
        while(communication.getConnection()->getStatus() != PAHOMQTTConnectionStatus::CONNECTED){
            sleep(1);
        }
        VO vo(config, &tlmData, &communication);
        vo.run();
    }
    else{
        VO vo(config);
        vo.run();
    }

    return 0;
}