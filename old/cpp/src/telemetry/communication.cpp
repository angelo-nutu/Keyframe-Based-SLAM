#include <utility>

#include "communication.hpp"
#include "data.h"
#include "connection.h"
#include "paho_mqtt_connection.hpp"
#include "raylib/src/raylib.h"
#include "mqtt_topics.h"
#include <sys/time.h>



Communication::Communication(const std::string &host,
                             const std::string &vehicleId, TelemetryData *data)
    : vehicleId(vehicleId) {
  this->data = data;

  PAHOMQTTConnectionParameters parameters =
      PAHOMQTTConnectionParameters::get_localhost_default();
  parameters.uri = host;
  connection = PAHOMQTTConnection(parameters);
  this->data->connection = &this->connection;

  connection.setUserData(this);

  connection.setOnConnectCallback(onConnectCallback);
  connection.setOnDisconnectCallback(onDisconnectCallback);
  connection.setOnErrorCallback(onErrorCallback);
  connection.setOnMessageCallback(onMessageCallback);
  connection.connect();
}

void Communication::onConnectCallback(PAHOMQTTConnection *connection,
                                      void *userData) {
  std::cout << "Connected" << std::endl;
  auto *tlmData = (Communication *)userData;
  auto topicVehicleState = MQTTTopics::GetTopicExtraTlmDataVehicleState(tlmData->vehicleId, "onboard");
  auto topicGPSMapsOrigins = MQTTTopics::GetTopicExtraTlmDataGpsMapOrigins(tlmData->vehicleId, "onboard");
  connection->subscribe(topicVehicleState.topic, topicVehicleState.qos);
  connection->subscribe(topicGPSMapsOrigins.topic, topicGPSMapsOrigins.qos);
}

void Communication::onDisconnectCallback(PAHOMQTTConnection *connection,
                                         void *userData) {
  std::cout << "Disconnected" << std::endl;
}

void Communication::onErrorCallback(PAHOMQTTConnection *connection,
                                    void *userData, const mqtt::token &tok) {
  std::cout << "Error: " << tok.get_error_message() << std::endl;
}

void Communication::onMessageCallback(PAHOMQTTConnection *connection,
                                      void *userData,
                                      const PAHOMQTTMessage &message) {
  auto *tlmData = (Communication *)userData;
  if (message.getTopic() == MQTTTopics::GetTopicExtraTlmDataVehicleState(tlmData->vehicleId, "onboard").topic) {

    if (tlmData->data->start){
      tlmData->data->vehicleState.deserializeFromJsonString(message.getPayload());
      
      if(tlmData->data->create_rotoTranMatrix){
        shared_mutex.lock();
        tlmData->data->create_rotoTranMatrix = false;
        double heading = tlmData->data->vehicleState.heading;
        double yaw = (180 - cv::abs(heading)) * ((heading > 0 ? +1 : -1));
        tlmData->data->rotoTranMat = (cv::Mat_<double>(4, 4) << 
           std::cos(yaw),    std::sin(yaw),  0, tlmData->data->vehicleState.x,  
                       0,                0, -1, tlmData->data->vehicleState.y,  
           std::sin(yaw),   -std::cos(yaw),  0,                             0,
                       0,                0,  0,                             1
          );
        tlmData->data->reset_VO = true;
        
        // auto topicVehicleState = MQTTTopics::GetTopicExtraTlmDataVehicleState(tlmData->vehicleId, "onboard");
        // connection->unsubscribe(topicVehicleState.topic);

        shared_mutex.unlock();
        
      }
    }

  } else if (message.getTopic() == MQTTTopics::GetTopicExtraTlmDataGpsMapOrigins(tlmData->vehicleId, "onboard").topic) {
    tlmData->data->gpsMapOrigins.deserializeFromProtobufString(message.getPayload());

    auto it = tlmData->data->gpsMapOrigins.origins.find(tlmData->data->gpsMapOrigins.trackLocation);
    if (it != tlmData->data->gpsMapOrigins.origins.end()) {
        GPSMapOrigin new_origin = it->second;
        
        if (new_origin.altitude != tlmData->data->current_origin.altitude || 
            new_origin.latitude != tlmData->data->current_origin.latitude || 
            new_origin.longitude != tlmData->data->current_origin.longitude) {
            
              shared_mutex.lock();
              tlmData->data->current_origin.altitude = new_origin.altitude;
              tlmData->data->current_origin.latitude = new_origin.latitude;
              tlmData->data->current_origin.longitude = new_origin.longitude;
                
              tlmData->data->create_rotoTranMatrix = true;

              std::cout << std::endl << "#################################################" << std::endl 
                  << "NEW GPS ORIGIN RECEIVED " << std::endl
                  << "altitude: " << tlmData->data->current_origin.altitude << std::endl 
                  << "latitude: " << tlmData->data->current_origin.latitude << std::endl 
                  << "longitude: " << tlmData->data->current_origin.longitude << std::endl
                  << "#################################################" << std::endl << std::endl; 
              shared_mutex.unlock();
        }
        

    } else {
      std::cerr << std::endl << "#################################################" << std::endl 
          << "GPS KEY NOT FOUND: " << tlmData->data->gpsMapOrigins.trackLocation << std::endl
          << "#################################################" << std::endl << std::endl;
    }

  }
}

bool Communication::sendCoordinates(double x, double y) {
  Serializers::Data::ValuesMap valuesMap;
  valuesMap.timestamp.values.push_back(getTimestampMicroseconds());
  valuesMap.valuesMap["x"].values.push_back(x);
  valuesMap.valuesMap["y"].values.push_back(y);
  Serializers::Data::TimeValuesPack valuesPack;
  valuesPack.valuesPack["VISUAL_ODOMETRY"] = valuesMap;
  if (this->connection.getStatus() == PAHOMQTTConnectionStatus::CONNECTED) {
    return this->connection.send(
        PAHOMQTTMessage(MQTTTopics::GetTopicExtraDataToLog(this->vehicleId, "onboard").topic, valuesPack.serializeAsProtobufString()));
  }
  return false;
}

static uint64_t getTimestampMicroseconds() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}