#include <queue>

//%deps(opencv)
#include <opencv2/opencv.hpp>

#include "camera/camera_image_message.hh"
#include "embedded/imu_driver/imu_message.hh"
#include "infrastructure/logging/log_reader.hh"

//%deps(fiducial_pose)
//%deps(rotation_to)
#include "third_party/experiments/estimation/vision/fiducial_pose.hh"
#include "third_party/experiments/geometry/rotation_to.hh"

//%deps(jet_filter)
//%deps(simple_geometry)
//%deps(jet_optimizer)
//%deps(window_3d)
#include "third_party/experiments/estimation/jet/jet_filter.hh"
#include "third_party/experiments/estimation/jet/jet_optimizer.hh"
#include "third_party/experiments/viewer/primitives/simple_geometry.hh"
#include "third_party/experiments/viewer/window_3d.hh"

//%deps(plot)
#include "third_party/experiments/viewer/primitives/plot.hh"

//%deps(time_interpolator)
#include "third_party/experiments/geometry/spatial/time_interpolator.hh"

//%deps(fit_ellipse)
#include "third_party/experiments/geometry/shapes/fit_ellipse.hh"

//
#include "vision/fiducial_detection_and_pose.hh"
#include "vision/fiducial_detection_message.hh"

namespace jet {
namespace filtering {
namespace {
using estimation::jet_filter::get_world_from_body;
namespace ejf = estimation::jet_filter;

void draw_states(viewer::SimpleGeometry& geo,
                 const std::vector<ejf::JetOptimizer::StateObservation>& states,
                 bool truth) {
  const int n_states = static_cast<int>(states.size());
  for (int k = 0; k < n_states; ++k) {
    auto& state = states.at(k).x;
    const SE3 T_world_from_body = get_world_from_body(state);
    if (truth) {
      geo.add_axes({T_world_from_body, 0.1});
    } else {
      geo.add_axes({T_world_from_body, 0.05, 2.0, true});

      const jcc::Vec3 velocity_world_frame = state.eps_dot.head<3>();
      geo.add_line({T_world_from_body.translation(), T_world_from_body.translation() + velocity_world_frame,
                    jcc::Vec4(0.7, 0.3, 0.3, 0.8)});

      if (k < n_states - 1) {
        const auto& next_state = states.at(k + 1).x;
        const SE3 T_world_from_body_next = get_world_from_body(next_state);
        geo.add_line(
            {T_world_from_body.translation(), T_world_from_body_next.translation(), jcc::Vec4(0.8, 0.8, 0.8, 0.4)});
      }
    }
  }
}

void plot_states(viewer::Plot& plot,
                 const ejf::JetPoseOptimizer::Solution& soln,
                 const estimation::TimePoint& first_t) {
  viewer::LinePlot accels_plot;

  accels_plot.subplots["opt_est_x"].color = jcc::Vec4(1.0, 0.0, 0.0, 0.4);
  accels_plot.subplots["opt_est_x"].line_width = 1.0;
  accels_plot.subplots["opt_est_y"].color = jcc::Vec4(0.0, 1.0, 0.0, 0.4);
  accels_plot.subplots["opt_est_y"].line_width = 1.0;
  accels_plot.subplots["opt_est_z"].color = jcc::Vec4(0.0, 0.0, 1.0, 0.4);
  accels_plot.subplots["opt_est_z"].line_width = 1.0;

  accels_plot.subplots["opt_est_ddot_x"].color = jcc::Vec4(1.0, 0.0, 0.0, 0.4);
  accels_plot.subplots["opt_est_ddot_x"].line_width = 1.0;
  accels_plot.subplots["opt_est_ddot_x"].dotted = true;
  accels_plot.subplots["opt_est_ddot_y"].color = jcc::Vec4(0.0, 1.0, 0.0, 0.4);
  accels_plot.subplots["opt_est_ddot_y"].line_width = 1.0;
  accels_plot.subplots["opt_est_ddot_y"].dotted = true;
  accels_plot.subplots["opt_est_ddot_z"].color = jcc::Vec4(0.0, 0.0, 1.0, 0.4);
  accels_plot.subplots["opt_est_ddot_z"].line_width = 1.0;
  accels_plot.subplots["opt_est_ddot_z"].dotted = true;

  for (const auto& x : soln.x) {
    const jcc::Vec3 optimized_accel = observe_accel(x.x, soln.p).observed_acceleration;
    accels_plot.subplots["opt_est_x"].points.push_back(
        {estimation::to_seconds(x.time_of_validity - first_t), optimized_accel.x()});
    accels_plot.subplots["opt_est_y"].points.push_back(
        {estimation::to_seconds(x.time_of_validity - first_t), optimized_accel.y()});
    accels_plot.subplots["opt_est_z"].points.push_back(
        {estimation::to_seconds(x.time_of_validity - first_t), optimized_accel.z()});

    accels_plot.subplots["opt_est_ddot_x"].points.push_back(
        {estimation::to_seconds(x.time_of_validity - first_t), x.x.eps_ddot.head<3>().x()});
    accels_plot.subplots["opt_est_ddot_y"].points.push_back(
        {estimation::to_seconds(x.time_of_validity - first_t), x.x.eps_ddot.head<3>().y()});
    accels_plot.subplots["opt_est_ddot_z"].points.push_back(
        {estimation::to_seconds(x.time_of_validity - first_t), x.x.eps_ddot.head<3>().z()});
  }

  plot.add_line_plot(accels_plot);
}

}  // namespace

void setup() {
  const auto view = viewer::get_window3d("Filter Debug");
  view->set_target_from_world(SE3(SO3::exp(Eigen::Vector3d(-3.1415 * 0.5, 0.0, 0.0)), jcc::Vec3(-1.0, 0.0, -1.0)));
  view->set_continue_time_ms(10);
  const auto background = view->add_primitive<viewer::SimpleGeometry>();
  const geometry::shapes::Plane ground{jcc::Vec3::UnitZ(), 0.0};
  background->add_plane({ground, 0.1});
  background->flip();
}

estimation::TimePoint to_time_point(const Timestamp& ts) {
  const auto epoch_offset = std::chrono::nanoseconds(uint64_t(ts));
  const estimation::TimePoint time_point = estimation::TimePoint{} + epoch_offset;
  return time_point;
}

class Calibrator {
 public:
  Calibrator() {
    const auto view = viewer::get_window3d("Filter Debug");
    geo_ = view->add_primitive<viewer::SimpleGeometry>();
  }

  void maybe_add_imu(const ImuMessage& msg) {
    const auto time_of_validity = to_time_point(msg.timestamp);
    if (estimation::to_seconds(time_of_validity - earliest_camera_time_) > 25.0) {
      return;
    }
    // if (timestamp > earliest_camera_time_) {
    if (true) {
      // std::cout << "Accel:"  << uint64_t(msg.timestamp) << std::endl;
      const jcc::Vec3 accel_mpss(msg.accel_mpss_x, msg.accel_mpss_y, msg.accel_mpss_z);

      ejf::AccelMeasurement accel_meas;
      accel_meas.observed_acceleration = accel_mpss;
      accel_meas_.push_back({accel_meas, time_of_validity});

      const jcc::Vec3 gyro_radps(msg.gyro_radps_x, msg.gyro_radps_y, msg.gyro_radps_z);
      ejf::GyroMeasurement gyro_meas;
      gyro_meas.observed_w = gyro_radps;
      gyro_meas_.push_back({gyro_meas, time_of_validity + estimation::to_duration(0.000001)});

      const jcc::Vec3 mag_utesla(msg.mag_utesla_x, msg.mag_utesla_y, msg.mag_utesla_z);
      mag_utesla_.push_back({mag_utesla, time_of_validity});
      geo_->add_point({(mag_utesla)});
      geo_->flush();
    }
    // Otherwise, ignore it
  }

  void add_fiducial(const Timestamp& ts, const SE3& world_from_camera) {
    const auto time_of_validity = to_time_point(ts);

    ejf::FiducialMeasurement fiducial_meas;
    fiducial_meas.T_fiducial_from_camera = world_from_camera;

    fiducial_meas_.push_back({fiducial_meas, time_of_validity});

    geo_->add_axes({world_from_camera, 0.025, 3.0});

    const auto view = viewer::get_window3d("Filter Debug");

    geo_->flush();
  }

  geometry::spatial::TimeInterpolator fit_mag() const {
    std::vector<jcc::Vec3> measurements;
    const auto view = viewer::get_window3d("Filter Debug");
    for (const auto& pt : mag_utesla_) {
      measurements.push_back(pt.first);
    }

    const auto ell_geo = view->add_primitive<viewer::SimpleGeometry>();
    const auto visitor = [&ell_geo, &view](const geometry::shapes::EllipseFit& fit) {
      ell_geo->add_ellipsoid({fit.ellipse, jcc::Vec4(0.4, 0.6, 0.4, 0.7), 2.0});
      ell_geo->flip();
    };
    const auto result = geometry::shapes::fit_ellipse(measurements, visitor);

    ell_geo->add_ellipsoid({result.ellipse, jcc::Vec4(0.2, 1.0, 0.2, 1.0), 4.0});
    ell_geo->flip();
    view->spin_until_step();

    std::vector<geometry::spatial::TimeControlPoint> control_points;
    for (const auto& pt : mag_utesla_) {
      const auto pt_time = pt.second;

      const jcc::Vec3 pt_mag_utesla = pt.first;

      const jcc::Vec3 pt_mag_corrected =
          (result.ellipse.cholesky_factor.transpose().inverse() * (pt_mag_utesla - result.ellipse.p0));

      ell_geo->add_point({pt_mag_corrected, jcc::Vec4(1.0, 0.2, 0.3, 1.0)});

      control_points.push_back({pt_time, pt_mag_corrected});
    }

    ell_geo->flip();

    const geometry::spatial::TimeInterpolator interpolator(control_points);
    return interpolator;
  }

  geometry::spatial::TimeInterpolator make_accel_interpolator() const {
    std::vector<geometry::spatial::TimeControlPoint> points;
    const auto view = viewer::get_window3d("Filter Debug");
    const auto accel_geo = view->add_primitive<viewer::SimpleGeometry>();

    for (const auto& measurement : accel_meas_) {
      accel_geo->add_point({measurement.first.observed_acceleration, jcc::Vec4(0.1, 0.7, 0.3, 0.9)});
      points.push_back({measurement.second, measurement.first.observed_acceleration});
    }

    accel_geo->flush();

    const geometry::spatial::TimeInterpolator interp(points);
    return interp;
  }

  geometry::spatial::TimeInterpolator make_gyro_interpolator() const {
    std::vector<geometry::spatial::TimeControlPoint> points;
    for (const auto& measurement : gyro_meas_) {
      points.push_back({measurement.second, measurement.first.observed_w});
    }
    const geometry::spatial::TimeInterpolator interp(points);
    return interp;
  }

  std::vector<ejf::JetOptimizer::StateObservation> test_filter() {
    const auto view = viewer::get_window3d("Filter Debug");
    const auto accel_interpolator = make_accel_interpolator();
    const auto gyro_interpolator = make_gyro_interpolator();

    const SE3 imu_from_vehicle = jf_.parameters().T_imu_from_vehicle;

    std::vector<ejf::JetOptimizer::StateObservation> est_states;
    geo_->clear();

    const auto timestep_geo = view->add_primitive<viewer::SimpleGeometry>();

    const auto plot_prim = view->add_primitive<viewer::Plot>();
    viewer::LinePlot accels_plot;

    const auto first_t = fiducial_meas_.front().second;
    for (const auto& fiducial_meas : fiducial_meas_) {
      const estimation::TimePoint t = fiducial_meas.second;
      if (t < first_t + estimation::to_duration(0.5)) {
        continue;
      }

      if (!got_camera_) {
        auto xp0 = ejf::JetFilter::reasonable_initial_state();

        const SE3 world_from_camera = fiducial_meas.first.T_fiducial_from_camera;
        xp0.x.x_world = world_from_camera.translation();
        xp0.x.R_world_from_body = world_from_camera.so3();

        xp0.x.accel_bias = jcc::Vec3(-0.0820206, 0.130374, -0.0765352);
        xp0.x.eps_ddot.head<3>() = jcc::Vec3(-0.0311536, 0.00954281, 0.00866376);
        xp0.x.eps_dot.head<3>() = jcc::Vec3(0.0668085, -0.179568, 0.0647334);
        xp0.x.eps_dot.tail<3>() = jcc::Vec3(0.0105246, 0.0498496, 0.0707742);

        xp0.time_of_validity = t;
        jf_.reset(xp0);
        got_camera_ = true;

        earliest_camera_time_ = t;
      }

      jf_.measure_fiducial(fiducial_meas.first, t);
      jet_opt_.measure_fiducial(fiducial_meas.first, t);

      std::cout << "Camera Location: " << fiducial_meas.first.T_fiducial_from_camera.translation().transpose()
                << std::endl;

      geo_->add_axes({fiducial_meas.first.T_fiducial_from_camera, 0.001, 3.0});
    }

    // for (const auto& accel_meas : accel_meas_) {
    for (std::size_t k = 0; k < accel_meas_.size(); ++k) {
      const auto& accel_meas = accel_meas_.at(k);
      const auto& gyro_meas = gyro_meas_.at(k);

      const estimation::TimePoint t = accel_meas.second;

      MatNd<3, 3> L;
      // clang-format off
      {
        L.row(0) << 9.67735, 0, 0;
        L.row(1) << 0.136597, 9.59653, 0;
        L.row(2) << -0.216635, 0.00400047, 9.64812;
      }  // clang-format on
      const jcc::Vec3 offset(0.0562102, 0.42847, -0.119841);
      const geometry::shapes::Ellipse ellipse{L, offset};
      const VecNd<3> compensated_accel =
          geometry::shapes::deform_ellipse_to_unit_sphere(accel_meas.first.observed_acceleration, ellipse) * 9.81;

      accels_plot.subplots["true_x"].points.push_back({estimation::to_seconds(t - first_t), compensated_accel.x()});
      accels_plot.subplots["true_y"].points.push_back({estimation::to_seconds(t - first_t), compensated_accel.y()});
      accels_plot.subplots["true_z"].points.push_back({estimation::to_seconds(t - first_t), compensated_accel.z()});

      if (t <= (first_t + estimation::to_duration(1.5))) {
        std::cout << "Skipping" << std::endl;
        continue;
      }

      jf_.measure_imu({compensated_accel}, t);
      jet_opt_.measure_imu({compensated_accel}, t);

      // jf_.measure_imu(accel_meas.first, t);
      // jet_opt_.measure_imu(accel_meas.first, t);

      // jf_.measure_gyro(gyro_meas.first, t + estimation::to_duration(0.000001));
      // jet_opt_.measure_gyro(gyro_meas.first, t);
    }

    accels_plot.subplots["true_x"].color = jcc::Vec4(1.0, 0.0, 0.0, 0.8);
    accels_plot.subplots["true_x"].line_width = 4.0;
    accels_plot.subplots["true_y"].color = jcc::Vec4(0.0, 1.0, 0.0, 0.8);
    accels_plot.subplots["true_y"].line_width = 4.0;
    accels_plot.subplots["true_z"].color = jcc::Vec4(0.0, 0.0, 1.0, 0.8);
    accels_plot.subplots["true_z"].line_width = 4.0;

    accels_plot.subplots["est_x"].color = jcc::Vec4(1.0, 0.0, 0.0, 0.4);
    accels_plot.subplots["est_x"].line_width = 1.0;
    accels_plot.subplots["est_y"].color = jcc::Vec4(0.0, 1.0, 0.0, 0.4);
    accels_plot.subplots["est_y"].line_width = 1.0;
    accels_plot.subplots["est_z"].color = jcc::Vec4(0.0, 0.0, 1.0, 0.4);
    accels_plot.subplots["est_z"].line_width = 1.0;

    estimation::TimePoint prev_time;

    while (true) {
      std::cout << "\n" << std::endl;
      const auto maybe_state = jf_.next_measurement();
      if (!maybe_state) {
        break;
      }

      const auto state = jf_.state().x;
      est_states.push_back({state, jf_.state().time_of_validity});

      const auto cov = jf_.state().P;

      const auto t = jf_.state().time_of_validity;

      std::cout << "dt: " << estimation::to_seconds(t - prev_time) << std::endl;
      prev_time = t;

      const SE3 T_world_from_body = get_world_from_body(state);

      constexpr double M_PER_MPSS = 0.01;

      const jcc::Vec3 meas_accel_imu_t = *accel_interpolator(t);
      const jcc::Vec3 expected_accel_imu = observe_accel(state, jf_.parameters()).observed_acceleration;

      const jcc::Vec3 meas_gyro_imu_t = *gyro_interpolator(t);
      const jcc::Vec3 expected_gyro_imu = observe_gyro(state, jf_.parameters()).observed_w;
      /*
            geo_->add_line({T_world_from_body.translation(), T_world_from_body * (meas_accel_imu_t * M_PER_MPSS),
                            jcc::Vec4(1.0, 0.0, 0.0, 0.8)});

            geo_->add_line({T_world_from_body.translation(), T_world_from_body * (expected_accel_imu * M_PER_MPSS),
                            jcc::Vec4(0.0, 1.0, 0.0, 0.8)});
      */
      const jcc::Vec3 g_world = jcc::Vec3::UnitZ() * 9.81;
      const jcc::Vec3 g_vehicle_frame = (T_world_from_body.so3().inverse() * g_world);

      // const jcc::Vec3 accel_g_subtracted_vehicle = meas_accel_imu_t - g_vehicle_frame;
      const jcc::Vec3 g_imu = imu_from_vehicle.so3() * T_world_from_body.so3().inverse() * g_world;

      std::cout << "\n" << std::endl;
      std::cout << "Accelerometer: " << std::endl;
      std::cout << "\tg_imu:      " << g_imu.transpose() << std::endl;
      std::cout << "\tmeas accel: " << meas_accel_imu_t.transpose() << std::endl;
      std::cout << "\texp accel:  " << expected_accel_imu.transpose() << std::endl;

      std::cout << "Specific Force:" << std::endl;
      std::cout << "\tmeas sp f: " << (meas_accel_imu_t - g_imu).transpose() << std::endl;
      std::cout << "\texp sp f: " << (expected_accel_imu - g_imu).transpose() << std::endl;

      std::cout << "States:       " << std::endl;
      std::cout << "\taccel_bias: " << state.accel_bias.transpose() << std::endl;
      std::cout << "\tgyro_bias: " << state.gyro_bias.transpose() << std::endl;
      std::cout << "\teps_ddot:   " << state.eps_ddot.transpose() << std::endl;
      std::cout << "\teps_dot:    " << state.eps_dot.transpose() << std::endl;
      std::cout << "\tx:          " << T_world_from_body.translation().transpose() << std::endl;

      std::cout << "Gyroscope:    " << std::endl;
      std::cout << "\tmeas gyro:  " << meas_gyro_imu_t.transpose() << std::endl;
      std::cout << "\texp gyro:   " << expected_gyro_imu.transpose() << std::endl;

      geo_->add_axes({T_world_from_body, 0.01, 1.0, false});

      // const Eigen::LLT<MatNd<3, 3>> P_llt(
      // cov.block<3, 3>(ejf::StateDelta::x_world_error_ind, ejf::StateDelta::x_world_error_ind));
      // geo_->add_ellipsoid({geometry::shapes::Ellipse{P_llt.matrixL(), T_world_from_body.translation()}});
      const SE3 T_imu_from_vehicle = jf_.parameters().T_imu_from_vehicle;

      const jcc::Vec3 velocity_world_frame = state.eps_dot.head<3>();
      timestep_geo->add_line({T_world_from_body.translation(), T_world_from_body.translation() + velocity_world_frame,
                              jcc::Vec4(0.7, 0.3, 0.3, 0.8)});

      const jcc::Vec3 accel_world_frame = state.eps_ddot.head<3>();
      timestep_geo->add_line({T_world_from_body.translation(), T_world_from_body.translation() + accel_world_frame,
                              jcc::Vec4(0.3, 0.7, 0.7, 0.5)});

      const jcc::Vec3 meas_sp_f_world =
          (T_world_from_body * T_imu_from_vehicle.inverse()).so3() * (meas_accel_imu_t - g_imu);
      std::cout << "Meas sp f w; " << meas_sp_f_world.transpose() << std::endl;
      // timestep_geo->add_line({T_world_from_body.translation(), T_world_from_body.translation() + meas_sp_f_world,
      //                         jcc::Vec4(0.3, 0.7, 0.7, 0.8)});

      // accels_plot.subplots["est_x"].points.push_back({estimation::to_seconds(t - first_t), expected_accel_imu.x()});
      // accels_plot.subplots["est_y"].points.push_back({estimation::to_seconds(t - first_t), expected_accel_imu.y()});
      // accels_plot.subplots["est_z"].points.push_back({estimation::to_seconds(t - first_t), expected_accel_imu.z()});

      const jcc::Vec3 meas_accel_world = (T_world_from_body * T_imu_from_vehicle.inverse()).so3() * meas_accel_imu_t;
      timestep_geo->add_line({T_world_from_body.translation(), T_world_from_body.translation() + meas_accel_world,
                              jcc::Vec4(0.3, 0.7, 0.7, 0.8)});

      jcc::Vec3 prev_pos = T_world_from_body.translation();
      for (double added_t = 0.01; added_t < 0.2; added_t += 0.01) {
        const auto predicted_state = jf_.predict(t + estimation::to_duration(added_t));
        const jcc::Vec3 pos = get_world_from_body(predicted_state).translation();
        timestep_geo->add_line({prev_pos, pos});
        timestep_geo->add_sphere({pos, 0.001});
        prev_pos = pos;
      }

      geo_->flush();
      timestep_geo->flip();
      view->spin_until_step();
    }

    plot_prim->add_line_plot(accels_plot);
    view->spin_until_step();
    return est_states;
  }

  void run() {
    // prepare();
    const std::vector<ejf::JetOptimizer::StateObservation> est_states = test_filter();

    const auto view = viewer::get_window3d("Filter Debug");

    std::cout << "Calibrating" << std::endl;
    const auto visitor = make_visitor();
    const auto solution = jet_opt_.solve(est_states, jf_.parameters(), visitor);

    const auto& x0 = solution.x.at(0).x;

    std::cout << "States:       " << std::endl;
    std::cout << "\taccel_bias: " << x0.accel_bias.transpose() << std::endl;
    std::cout << "\tgyro_bias: " << x0.gyro_bias.transpose() << std::endl;
    std::cout << "\teps_ddot:   " << x0.eps_ddot.transpose() << std::endl;
    std::cout << "\teps_dot:    " << x0.eps_dot.transpose() << std::endl;
    std::cout << "\tx:          " << get_world_from_body(x0).translation().transpose() << std::endl;
    std::cout << "\tlog(r)      " << x0.R_world_from_body.log().transpose() << std::endl;
  }

 private:
  ejf::JetPoseOptimizer::Visitor make_visitor() {
    const auto view = viewer::get_window3d("Filter Debug");
    const auto visitor_geo = view->add_primitive<viewer::SimpleGeometry>();
    const auto plot_prim = view->add_primitive<viewer::Plot>();

    const auto first_t = fiducial_meas_.front().second;
    const auto visitor = [first_t, view, visitor_geo, plot_prim](const ejf::JetPoseOptimizer::Solution& soln) {
      plot_prim->clear();
      visitor_geo->clear();
      draw_states(*visitor_geo, soln.x, false);
      visitor_geo->flip();
      // std::cout << "\tOptimized g: " << soln.p.g_world.transpose() << std::endl;
      std::cout << "\tOptimized T_imu_from_vehicle: " << soln.p.T_imu_from_vehicle.translation().transpose() << "; "
                << soln.p.T_imu_from_vehicle.so3().log().transpose() << std::endl;

      plot_states(*plot_prim, soln, first_t);

      view->spin_until_step();
    };
    return visitor;
  }

  estimation::TimePoint earliest_camera_time_ = estimation::TimePoint::max();
  bool got_camera_ = false;

  std::vector<std::pair<ejf::AccelMeasurement, estimation::TimePoint>> accel_meas_;
  std::vector<std::pair<ejf::FiducialMeasurement, estimation::TimePoint>> fiducial_meas_;
  std::vector<std::pair<ejf::GyroMeasurement, estimation::TimePoint>> gyro_meas_;

  std::vector<std::pair<jcc::Vec3, estimation::TimePoint>> mag_utesla_;

  ejf::JetFilter jf_;
  ejf::JetOptimizer jet_opt_;

  // temp
  std::shared_ptr<viewer::SimpleGeometry> geo_;
};

void go() {
  setup();
  const auto view = viewer::get_window3d("Filter Debug");
  const auto geo = view->add_primitive<viewer::SimpleGeometry>();
  const std::vector<std::string> channel_names = {"imu", "fiducial_detection_channel", "camera_image_channel"};

  // const std::string path = "/jet/logs/calibration-log-jan26-1";
  // const std::string path = "/jet/logs/calibration-log-jan31-1";
  // const std::string path = "/jet/logs/calibration-log-feb9-1";
  // const std::string path = "/jet/logs/calibration-log-feb9-2";
  const std::string path = "/jet/logs/calibration-log-feb14-1";
  // const std::string path = "/jet/logs/imu-calibration-log-feb-17-1";

  Calibrator calibrator;
  jet::LogReader reader(path, channel_names);

  bool accepted_any = false;
  SE3 last_world_from_camera;

  constexpr bool USE_CAMERA_IMAGES = false;
  constexpr bool USE_FIDUCIAL_DETECTIONS = true;

  int imu_ct = 0;
  for (int k = 0; k < 3000; ++k) {
    {
      ImuMessage imu_msg;
      if (reader.read_next_message("imu", imu_msg)) {
        imu_ct++;
        // if (imu_ct % 1 == 0) {
        if (imu_ct) {
          calibrator.maybe_add_imu(imu_msg);
        }
      } else {
        std::cout << "Breaking at : " << k << std::endl;
        break;
      }
    }

    if (USE_CAMERA_IMAGES) {
      CameraImageMessage cam_msg;
      if (reader.read_next_message("camera_image_channel", cam_msg)) {
        const auto image = get_image_mat(cam_msg);

        const auto result = detect_board(image);
        if (result) {
          const SE3 world_from_camera = *result;

          if (accepted_any) {
            const SE3 camera_from_last_camera = world_from_camera.inverse() * last_world_from_camera;
            constexpr double MAX_OUTLIER_DIST_M = 0.7;
            if (camera_from_last_camera.translation().norm() > MAX_OUTLIER_DIST_M) {
              continue;
            }
          }

          accepted_any = true;
          last_world_from_camera = world_from_camera;

          calibrator.add_fiducial(cam_msg.timestamp, world_from_camera);
        }
      }
    }

    if (USE_FIDUCIAL_DETECTIONS) {
      FiducialDetectionMessage fiducial_msg;
      if (reader.read_next_message("fiducial_detection_channel", fiducial_msg)) {
        const SE3 world_from_camera = fiducial_msg.fiducial_from_camera();

        // if (accepted_any) {
        //   const SE3 camera_from_last_camera = world_from_camera.inverse() * last_world_from_camera;
        //   constexpr double MAX_OUTLIER_DIST_M = 0.7;
        //   if (camera_from_last_camera.translation().norm() > MAX_OUTLIER_DIST_M) {
        //     continue;
        //   }
        // }

        calibrator.add_fiducial(fiducial_msg.timestamp, world_from_camera);
        accepted_any = true;
        last_world_from_camera = world_from_camera;
      }
    }
    // cv::imshow("Image", image);
    // cv::waitKey(0);
  }

  geo->flush();
  view->spin_until_step();
  std::cout << "Done, preparing to calibrate" << std::endl;

  calibrator.run();
  view->spin_until_step();
}

}  // namespace filtering
}  // namespace jet

int main() {
  jet::filtering::go();
}
