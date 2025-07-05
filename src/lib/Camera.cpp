#include "Camera.hpp"

Camera::Camera() :
    bBoth(false),
    zmqContext(1),
    zmqSubscriber(zmqContext, zmq::socket_type::sub) {
    
    std::string protocol = "tcp";
    std::string port = "5555";
    std::string topic = "camera/rgbd";

    this->bRunThread = true;
    
    thrCapture = std::thread(&Camera::Capture, this, protocol, port, topic);
}

Camera::~Camera(){
    this->bRunThread = false;

    if (thrCapture.joinable())
        thrCapture.join();
}

void Camera::Capture(std::string protocol, std::string port, std::string topic){
    this->zmqSubscriber.connect(protocol + "://localhost:" + port);
    this->zmqSubscriber.set(zmq::sockopt::subscribe, topic);
    LOG("Connected to localhost on port " + port + " at topic " + topic);

    while(this->bRunThread){
        zmq::message_t message;
        int part = 0; 

        while(true){
            this->zmqSubscriber.recv(message);
            if (message.empty()){
                ERROR("No message arrived!");
                INFO("Make sure the port is correct and the sender is active");
                {
                    std::lock_guard<std::mutex> lock(this->mtxImgs);
                    this->matRgbLast.release();
                    this->matDepthLast.release();
                    this->bBoth = true;
                }
                cvImgs.notify_one(); 
                break;
            }

            {
                std::lock_guard<std::mutex> lock(this->mtxImgs);
                if (part == 1) {
                    this->bBoth = false;
                    deserializeFromProto(message.to_string(), this->matRgbLast);
                } else if (part == 2) {
                    deserializeFromProto(message.to_string(), this->matDepthLast);
                    this->bBoth = true;
                    LOG("Captured images");
                }
            }
            if (part == 2) {
                cvImgs.notify_one(); 
            }

            if(!message.more()){
                break;
            }
            part++;
        }
    }
}

std::optional<std::pair<cv::Mat, cv::Mat>> Camera::GrabFrames(){
    std::unique_lock<std::mutex> lock(this->mtxImgs);
    cvImgs.wait(lock, [this]{ return this->bBoth; }); 

    if(this->matRgbLast.empty() || this->matDepthLast.empty()) {
        return std::nullopt;
    }

    this->bBoth = false;

    return std::make_pair(this->matRgbLast, this->matDepthLast);
}
