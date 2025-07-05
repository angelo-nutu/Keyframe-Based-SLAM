#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include "data.h"
#include "paho_mqtt_connection.hpp"
#include "mutex.hpp"

class Communication {
public:
  Communication() = default;
  Communication(const std::string &host, const std::string &mvehicleId,
                TelemetryData *data);
  ~Communication() = default;

  bool sendCoordinates(double x, double y);

  PAHOMQTTConnection *getConnection() { return &this->connection; };
  void setVehicleId(const std::string &_vehicleId) {
    this->vehicleId = _vehicleId;
  }
  const std::string &getVehicleId() { return this->vehicleId; }

private:
  TelemetryData *data;
  PAHOMQTTConnection connection;
  std::string vehicleId;

  // Callbacks
  static void onConnectCallback(PAHOMQTTConnection *connection, void *userData);
  static void onDisconnectCallback(PAHOMQTTConnection *connection,
                                   void *userData);
  static void onErrorCallback(PAHOMQTTConnection *connection, void *userData,
                              const mqtt::token &tok);
  static void onMessageCallback(PAHOMQTTConnection *connection, void *userData,
                                const PAHOMQTTMessage &message);
};
static uint64_t getTimestampMicroseconds();
#endif // COMMUNICATION_H