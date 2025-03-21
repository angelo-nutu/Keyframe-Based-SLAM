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
    tlmData->data->vehicleState.deserializeFromJsonString(message.getPayload());
  } else if (message.getTopic() == MQTTTopics::GetTopicExtraTlmDataGpsMapOrigins(tlmData->vehicleId, "onboard").topic) {
    tlmData->data->gpsMapOrigins.deserializeFromJsonString(message.getPayload());

    GPSMapOrigin new_origin = tlmData->data->gpsMapOrigins.origins.at(tlmData->data->gpsMapOrigins.trackLocation);
    if (new_origin.altitude != tlmData->data->current_origin.altitude || new_origin.latitude != tlmData->data->current_origin.latitude || new_origin.longitude != tlmData->data->current_origin.longitude) {

      double heading = tlmData->data->vehicleState.heading;
      tlmData->data->rotoTranMat = (cv::Mat_<double>(4, 4) << 
        -std::cos(heading),   0,  -std::sin(heading),  tlmData->data->vehicleState.x,  
                         0,   1,                   0,                              0,  
         std::sin(heading),   0,  -std::cos(heading),  tlmData->data->vehicleState.y, 
                         0,   0,                   0,                              1
        );
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