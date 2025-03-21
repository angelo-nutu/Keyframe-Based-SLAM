#include <vo.hpp>
#include <config.hpp>

int main(int argc, char** argv) {
    
    std::string path;
    
    if (argc > 1) {
        path = argv[1];
    } else {
        std::cerr << "No path provided for the configuration file." << std::endl;
        return 1;
    }

    Config config(path);

    VO vo(config);
    vo.run();

    return 0;
}