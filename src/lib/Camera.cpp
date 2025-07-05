#include "Camera.hpp"

Camera::Camera() :
    bAll(false),
    bKReady(false),
    zmqContext(1),
    zmqSubscriber(zmqContext, zmq::socket_type::sub) {
    
    std::string protocol = "tcp";
    std::string port = "5555";
    std::string topicRGBD = "camera/rgbd";
    std::string topicIntrinsics = "camera/intrinsics";

    this->bRunThread = true;
    
    thrCapture = std::thread(&Camera::Capture, this, protocol, port, topicRGBD, topicIntrinsics);
}

Camera::~Camera(){
    this->bRunThread = false;

    if (thrCapture.joinable())
        thrCapture.join();
}

void Camera::Capture(std::string protocol, std::string port, std::string topicRGBD, std::string topicIntrinsics){
    zmq::message_t message;

    this->zmqSubscriber.connect(protocol + "://localhost:" + port);

    this->zmqSubscriber.set(zmq::sockopt::subscribe, topicIntrinsics);
    INFO("Connected to localhost on port " + port + " at topic " + topicIntrinsics);
    this->zmqSubscriber.recv(message);
    this->zmqSubscriber.recv(message);
    if (message.empty()){
        ERROR("No intrinsics arrived!");
        FIX("Make sure the sender is processing data correctly");
        exit(EXIT_FAILURE);
    }
    
    {   
        std::lock_guard<std::mutex> lock(this->mtxImgs);
        deserializeFromProto(message.to_string(), this->itrK);
        this->bKReady = true;
        LOG("Received camera Intrinsics");
    }
    this->zmqSubscriber.set(zmq::sockopt::unsubscribe, topicIntrinsics);

    this->zmqSubscriber.set(zmq::sockopt::subscribe, topicRGBD);
    INFO("Connected to localhost on port " + port + " at topic " + topicRGBD);

    while(this->bRunThread){
        int part = 0; 

        while(true){
            this->zmqSubscriber.recv(message);
            if (message.empty()){
                ERROR("No message arrived!");
                FIX("Make sure the sender is processing data correctly");
                {
                    std::lock_guard<std::mutex> lock(this->mtxImgs);
                    this->matRgbLast.release();
                    this->matDepthLast.release();
                    this->bAll = true;
                }
                cvImgs.notify_one(); 
                break;
            }

            {
                std::lock_guard<std::mutex> lock(this->mtxImgs);
                if (part == 1) {
                    this->bAll = false;
                    deserializeFromProto(message.to_string(), this->matRgbLast);
                } else if (part == 2) {
                    deserializeFromProto(message.to_string(), this->matDepthLast);
                    this->bAll = false;
                } else if (part == 3) {
                    deserializeFromProto(message.to_string(), this->matMaskLast);
                    this->bAll = true;
                }
            }
            if (part == 3) {
                cvImgs.notify_one(); 
            }

            if(!message.more()){
                break;
            }
            part++;
        }
    }
}

std::optional<std::tuple<cv::Mat, cv::Mat, cv::Mat>> Camera::GrabFrames(){
    std::unique_lock<std::mutex> lock(this->mtxImgs);
    cvImgs.wait(lock, [this]{ return this->bAll; }); 

    if(this->matRgbLast.empty() || this->matDepthLast.empty() || this->matMaskLast.empty()) {
        return std::nullopt;
    }

    this->bAll = false;

    return std::make_tuple(this->matRgbLast.clone(), this->matDepthLast.clone(), this->matMaskLast.clone());
}

cv::Mat Camera::getK(){
    while(true){
        {   
            std::lock_guard<std::mutex> lock(this->mtxImgs);
            if (this->bKReady) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return  (cv::Mat_<double>(3,3) <<
                this->itrK.fx, 0,             this->itrK.ppx,
                0,             this->itrK.fy, this->itrK.ppy,
                0,             0,             1             );
}
