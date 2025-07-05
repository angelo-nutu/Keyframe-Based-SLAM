#ifndef DATA_H
#define DATA_H

#include "data/vehicle_state.h"
#include "paho_mqtt_connection.hpp"
#include "telemetry/gps_maps.h"
#include "data/extra.h"
#include <opencv2/opencv.hpp>

typedef Serializers::Data::VehicleState VehicleState;
typedef Serializers::Telemetry::GPSMapOrigins GPSMapOrigins;
typedef Serializers::Telemetry::GPSMapOrigin GPSMapOrigin;

struct TelemetryData {
  VehicleState vehicleState;
  GPSMapOrigins gpsMapOrigins;
  PAHOMQTTConnection *connection;

  GPSMapOrigin current_origin;
  cv::Mat rotoTranMat;
  bool start;
  bool create_rotoTranMatrix;
  bool reset_VO;
};

#endif // DATA_H