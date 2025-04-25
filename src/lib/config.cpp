#include <config.hpp>

Config::Config(){
    extraction = "orb";
    matching = "bf";
    pose = "pnp";

    accel_threshold = 0.3f;
    gyro_threshold = 0.6f;
    realtime = true;
    rosbag_path = "";

    yolo = false;
    model_path = "";

    telemetry = false;

    debug = true;
    display = true;
}

Config::Config(std::string& filename) {
    YAML::Node config = YAML::LoadFile(filename);

    extraction = config["odometry"]["extraction"].as<std::string>();
    matching = config["odometry"]["matching"].as<std::string>();
    pose = config["odometry"]["pose"].as<std::string>();

    accel_threshold = config["camera"]["accel-threshold"].as<float>();
    gyro_threshold = config["camera"]["gyro-threshold"].as<float>();
    realtime = config["camera"]["realtime"].as<bool>();
    rosbag_path = config["camera"]["path"].as<std::string>();

    yolo = config["yolo"]["enabled"].as<bool>();
    model_path = config["yolo"]["path"].as<std::string>();

    telemetry = config["telemetry"]["enabled"].as<bool>();
    host = config["telemetry"]["host"].as<std::string>();
    vehicleId = config["telemetry"]["vehicle_id"].as<std::string>();

    debug = config["debug"].as<bool>();
    display = config["display"].as<bool>();

    std::cout << "\nLoaded Configuration:\n";
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
    std::cout << std::left << std::setw(25) << "Telemetry Host:"
              << "\"" << host << "\"\n";
    std::cout << std::left << std::setw(25) << "Telemetry Vehicle ID:"
              << "\"" << vehicleId << "\"\n";
    std::cout << std::left << std::setw(25) << "Debug Enabled:" 
              << (debug ? "YES" : "NO") << "\n";
    std::cout << std::left << std::setw(25) << "Display Enabled:" 
              << (display ? "YES" : "NO") << "\n";
    std::cout << "----------------------------------------\n\n";
}