//%bin(camera_balsaq_main)
#include "camera/camera_balsaq.hh"
#include "infrastructure/balsa_queue/bq_main_macro.hh"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <sstream>

namespace jet {

constexpr double WEBCAM_EXPOSURE = 0.01;

void CameraBq::init(int argc, char *argv[]) {
  cap = cv::VideoCapture(0);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap.set(cv::CAP_PROP_FPS, CAMERA_FPS);
  // 0 is the id of video device.0 if you have only one camera.
  publisher_ = make_publisher("camera_image_channel");
}

void CameraBq::loop() {
  cv::Mat camera_frame;
  std::cout << "Camera BQ: trying to get a frame" << std::endl;

  cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
  cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
  cap.set(cv::CAP_PROP_EXPOSURE, WEBCAM_EXPOSURE);

  if (cap.read(camera_frame)) {
    // Get the image capture timestamp
    const long int cap_time_msec = cap.get(cv::CAP_PROP_POS_MSEC);
    const Timestamp cap_time_vehicle = msec_monotonic_to_vehicle_monotonic(cap_time_msec);

    std::cout << "T: " << cap_time_vehicle << std::endl;

    // Pack a message
    CameraImageMessage message;
    const std::size_t n_elements = camera_frame.rows * camera_frame.cols * 3u;
    message.image_data.resize(n_elements);
    constexpr std::size_t SIZE_OF_UCHAR = sizeof(uint8_t);
    if (camera_frame.isContinuous()) {
      std::memcpy(message.image_data.data(), camera_frame.data, SIZE_OF_UCHAR * n_elements);
    }
    message.timestamp = cap_time_vehicle;
    message.height = camera_frame.size().height;
    message.width = camera_frame.size().width;
    publisher_->publish(message);
    std::cout << "Camera BQ: publishes a camera frame " << message.width << " " << message.height << std::endl;
  } else {
  }
}
void CameraBq::shutdown() {
  std::cout << "Camera BQ: process shutting down." << std::endl;
}

Timestamp CameraBq::msec_monotonic_to_vehicle_monotonic(long int cap_time_msec) const {
  // Get the reported monotonic time in milliseconds
  const std::chrono::milliseconds cap_time_monotonic_msec(cap_time_msec);
  // Convert it to a timepoint
  const std::chrono::time_point<std::chrono::steady_clock> cap_time_monotonic(cap_time_monotonic_msec);

  // Get the current time in monotonic
  const auto cur_time_monotonic = std::chrono::steady_clock::now();
  // Get the current time in the vehicle wall clock
  const auto cur_time_vehicle = get_current_time().time_point();
  const auto cur_time_from_cap_time = cur_time_monotonic - cap_time_monotonic;
  const auto cap_time_vehicle = cur_time_vehicle - cur_time_from_cap_time;

  const auto wall_from_monotonic = (cur_time_vehicle.time_since_epoch()) - (cur_time_monotonic.time_since_epoch());

  std::cout << "Offset:              " << cur_time_from_cap_time.count() << std::endl;
  std::cout << "Cap msec:            " << cap_time_msec << std::endl;
  std::cout << "Monotonic:           " << cur_time_monotonic.time_since_epoch().count() << std::endl;
  std::cout << "Wall from monotonic: " << wall_from_monotonic.count() << std::endl;

  return Timestamp(cap_time_vehicle);
}

}  // namespace jet

BALSA_QUEUE_MAIN_FUNCTION(jet::CameraBq)
