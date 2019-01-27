
#pragma once

#include "camera/camera_image_message.hh"
#include "infrastructure/balsa_queue/balsa_queue.hh"
#include "infrastructure/comms/mqtt_comms_factory.hh"

//%deps(opencv)
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>

namespace jet {

class CameraBq : public BalsaQ {
 public:
  CameraBq() = default;
  void init(int argc, char *argv[]);
  void loop();
  void shutdown();

 private:
  // Use this to convert a capture time to a wall time
  //
  // @param msec_monotonic The timestamp in int milliseconds reported by "the" monotonic clock
  //
  // The camera capture times are reported in millseconds from monotonic clock start
  // This function reconstructs a a reasonable approximation of the *vehicle* time
  // associated with that monotonic time
  Timestamp msec_monotonic_to_vehicle_monotonic(long int msec_monotonic) const;
  static const int CAMERA_FPS = 10;
  PublisherPtr publisher_;
  cv::VideoCapture cap;
};

}  // namespace jet
