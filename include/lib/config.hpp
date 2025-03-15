#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <iostream>
#include <iomanip>
#include <cctype>
#include <string>
#include <yaml-cpp/yaml.h>
#include <raylib.h>


class Config{    
public:
    Config();
    Config(std::string& filename);

    std::string extraction;
    std::string matching;
    std::string pose;

    bool realtime;
    std::string rosbag_path;

    bool yolo;
    std::string model_path;

    bool baseline;
    std::string baseline_path;
private:

};

#endif
