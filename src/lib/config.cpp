#include <config.hpp>

Config::Config(){
    extraction = "orb";
    matching = "bf";
    pose = "pnp";

    realtime = true;
    rosbag_path = "";

    yolo = false;
    model_path = "";

    telemetry = false;
}

Config::Config(std::string& filename) {
    YAML::Node config = YAML::LoadFile(filename);

    extraction = config["odometry"]["extraction"].as<std::string>();
    matching = config["odometry"]["matching"].as<std::string>();
    pose = config["odometry"]["pose"].as<std::string>();

    realtime = config["camera"]["realtime"].as<bool>();
    rosbag_path = config["camera"]["path"].as<std::string>();

    yolo = config["yolo"]["enabled"].as<bool>();
    model_path = config["yolo"]["path"].as<std::string>();

    telemetry = config["telemetry"]["enabled"].as<bool>();

    std::cout << "Loaded Configuration:\n";
    std::cout << "----------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Odometry Extraction: ";
    std::transform(extraction.begin(), extraction.end(), std::ostream_iterator<char>(std::cout), ::toupper);
    std::cout << "\n";
    std::cout << std::left << std::setw(25) << "Odometry Matching: ";
    std::transform(matching.begin(), matching.end(), std::ostream_iterator<char>(std::cout), ::toupper);
    std::cout << "\n";
    std::cout << std::left << std::setw(25) << "Odometry Pose:" 
              << (pose == "pnp" ? "PnP" : (pose == "emat" ? "EMat" : pose)) << "\n";
    std::cout << std::left << std::setw(25) << "Camera Realtime:" 
              << (realtime ? "YES" : "NO") << "\n";
    std::cout << std::left << std::setw(25) << "Camera Path:" 
              << "\"" << rosbag_path << "\"\n";
    std::cout << std::left << std::setw(25) << "YOLO Enabled:" 
              << (yolo ? "YES" : "NO") << "\n";
    std::cout << std::left << std::setw(25) << "YOLO Model Path:" 
              << "\"" << model_path << "\"\n";
    std::cout << std::left << std::setw(25) << "Telemetry Enabled:" 
              << (telemetry ? "YES" : "NO") << "\n";
    std::cout << "----------------------------------------\n";
}