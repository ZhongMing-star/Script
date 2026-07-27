#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <functional>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>

class CameraCalibrationNode : public rclcpp::Node {
public:
  CameraCalibrationNode() : Node("camera_calibration_node") {
    image_topic_ = declare_parameter("image_topic", "/camera/image_raw");
    board_width_ = declare_parameter("board_width", 12);
    board_height_ = declare_parameter("board_height", 9);
    square_size_ = declare_parameter("square_size", 0.04);
    required_frames_ = declare_parameter("required_frames", 30);
    capture_interval_ = declare_parameter("capture_interval", 0.5);
    output_file_ = declare_parameter("output_file", std::string("camera_calibration.yaml"));
    captured_images_dir_ = declare_parameter("captured_images_dir", std::string("captured_images"));
    show_captured_image_ = declare_parameter("show_captured_image", true);
    if (show_captured_image_ && std::getenv("DISPLAY") == nullptr &&
        std::getenv("WAYLAND_DISPLAY") == nullptr) {
      show_captured_image_ = false;
      RCLCPP_WARN(
        get_logger(),
        "show_captured_image requested, but no DISPLAY or WAYLAND_DISPLAY is available; preview disabled");
    }
    last_capture_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    sub_ = create_subscription<sensor_msgs::msg::Image>(image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&CameraCalibrationNode::imageCallback, this, std::placeholders::_1));
    start_srv_ = create_service<std_srvs::srv::Trigger>("start_capture", std::bind(&CameraCalibrationNode::start, this, std::placeholders::_1, std::placeholders::_2));
    stop_srv_ = create_service<std_srvs::srv::Trigger>("stop_capture", std::bind(&CameraCalibrationNode::stop, this, std::placeholders::_1, std::placeholders::_2));
    clear_srv_ = create_service<std_srvs::srv::Trigger>("clear_samples", std::bind(&CameraCalibrationNode::clear, this, std::placeholders::_1, std::placeholders::_2));
    calibrate_srv_ = create_service<std_srvs::srv::Trigger>("calibrate", std::bind(&CameraCalibrationNode::calibrate, this, std::placeholders::_1, std::placeholders::_2));
    RCLCPP_INFO(
      get_logger(),
      "Listening on %s; board=%dx%d, required_frames=%d, capture_interval=%.3fs, captured_images_dir=%s",
      image_topic_.c_str(), board_width_, board_height_, required_frames_, capture_interval_,
      std::filesystem::absolute(captured_images_dir_).string().c_str());
  }
private:
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv::Mat image;
    try { image = cv_bridge::toCvShare(msg, "bgr8")->image; }
    catch (const cv_bridge::Exception &e) { RCLCPP_WARN(get_logger(), "Image conversion failed: %s", e.what()); return; }
    pumpPreviewEvents();
    std::lock_guard<std::mutex> lock(mutex_);
    if (!capturing_ || (this->now() - last_capture_).seconds() < capture_interval_) return;
    cv::Mat gray; cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> corners;
    cv::Size pattern(board_width_, board_height_);
    bool found = cv::findChessboardCorners(gray, pattern, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if (!found) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Capturing: chessboard not detected (samples=%zu/%d, expected_corners=%d)",
        image_points_.size(), required_frames_, board_width_ * board_height_);
      return;
    }
    cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.1));
    const auto sample_number = image_points_.size() + 1;
    const auto image_path = capturedImagePath(*msg, sample_number);
    cv::Mat annotated = image.clone();
    cv::drawChessboardCorners(annotated, pattern, corners, true);
    try {
      std::filesystem::create_directories(image_path.parent_path());
      if (!cv::imwrite(image_path.string(), annotated)) {
        RCLCPP_ERROR(get_logger(), "Detected chessboard but failed to save image: %s", image_path.string().c_str());
        return;
      }
    } catch (const std::exception &e) {
      RCLCPP_ERROR(
        get_logger(), "Detected chessboard but could not save image %s: %s",
        image_path.string().c_str(), e.what());
      return;
    }
    image_points_.push_back(corners); image_size_ = gray.size(); last_capture_ = now();
    showCapturedImage(annotated);
    RCLCPP_INFO(
      get_logger(), "Captured frame %zu/%d: corners=%zu, size=%dx%d, stamp=%d.%09u, saved=%s",
      image_points_.size(), required_frames_, corners.size(), image_size_.width, image_size_.height,
      msg->header.stamp.sec, msg->header.stamp.nanosec, image_path.string().c_str());
    if (static_cast<int>(image_points_.size()) >= required_frames_) { capturing_ = false; RCLCPP_INFO(get_logger(), "Enough samples collected; call /calibrate"); }
  }
  void pumpPreviewEvents() {
    if (!preview_window_open_) return;
    try {
      cv::waitKey(1);
    } catch (const cv::Exception &e) {
      RCLCPP_ERROR(get_logger(), "Failed to process captured image preview events: %s", e.what());
      show_captured_image_ = false;
      preview_window_open_ = false;
    }
  }
  void showCapturedImage(const cv::Mat &image) {
    if (!show_captured_image_) return;
    try {
      if (preview_window_open_) {
        cv::destroyWindow(preview_window_name_);
      }
      cv::namedWindow(preview_window_name_, cv::WINDOW_NORMAL);
      cv::imshow(preview_window_name_, image);
      cv::waitKey(1);
      preview_window_open_ = true;
    } catch (const cv::Exception &e) {
      RCLCPP_ERROR(get_logger(), "Failed to display captured image preview: %s", e.what());
      show_captured_image_ = false;
      preview_window_open_ = false;
    }
  }
  std::filesystem::path capturedImagePath(
    const sensor_msgs::msg::Image &msg, std::size_t sample_number) const {
    std::ostringstream name;
    name << "sample_" << std::setfill('0') << std::setw(3) << sample_number
         << "_" << msg.header.stamp.sec << "_" << std::setw(9) << msg.header.stamp.nanosec
         << ".png";
    return std::filesystem::path(captured_images_dir_) / name.str();
  }
  std::vector<cv::Point3f> objectPoints() const { std::vector<cv::Point3f> p; for(int y=0;y<board_height_;++y) for(int x=0;x<board_width_;++x) p.emplace_back(x*square_size_, y*square_size_, 0); return p; }
  void start(const std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr r) { std::lock_guard<std::mutex> l(mutex_); capturing_=true; r->success=true; r->message="capture started with " + std::to_string(image_points_.size()) + " existing samples"; RCLCPP_INFO(get_logger(), "%s", r->message.c_str()); }
  void stop(const std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr r) { std::lock_guard<std::mutex> l(mutex_); capturing_=false; r->success=true; r->message="capture stopped with " + std::to_string(image_points_.size()) + " samples"; RCLCPP_INFO(get_logger(), "%s", r->message.c_str()); }
  void clear(const std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr r) { std::lock_guard<std::mutex> l(mutex_); const auto count=image_points_.size(); image_points_.clear(); r->success=true; r->message=std::to_string(count) + " samples cleared; saved images retained in " + captured_images_dir_; RCLCPP_INFO(get_logger(), "%s", r->message.c_str()); }
  void calibrate(const std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr r) {
    std::lock_guard<std::mutex> l(mutex_); if(image_points_.size()<3){r->success=false;r->message="need at least 3 samples";return;}
    std::vector<std::vector<cv::Point3f>> obj(image_points_.size(), objectPoints()); cv::Mat camera, dist; std::vector<cv::Mat> rv,tv;
    double err=cv::calibrateCamera(obj,image_points_,image_size_,camera,dist,rv,tv); std::filesystem::path p(output_file_); if(p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::ofstream fs(output_file_); if(!fs){r->success=false;r->message="cannot open output file: "+output_file_;return;} fs<<std::setprecision(12);
    auto matrix_data=[&fs](const cv::Mat &m){ fs<<"["; for(int i=0;i<m.rows*m.cols;++i){if(i)fs<<", ";fs<<m.at<double>(i);} fs<<"]\n"; };
    fs<<"image_width: "<<image_size_.width<<"\nimage_height: "<<image_size_.height<<"\ncamera_name: calibrated_camera\n";
    fs<<"camera_matrix:\n  rows: 3\n  cols: 3\n  data: "; matrix_data(camera);
    fs<<"distortion_model: plumb_bob\ndistortion_coefficients:\n  rows: 1\n  cols: "<<dist.total()<<"\n  data: "; matrix_data(dist.reshape(1,1));
    fs<<"rectification_matrix:\n  rows: 3\n  cols: 3\n  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n";
    fs<<"projection_matrix:\n  rows: 3\n  cols: 4\n  data: ["<<camera.at<double>(0,0)<<", 0, "<<camera.at<double>(0,2)<<", 0, 0, "<<camera.at<double>(1,1)<<", "<<camera.at<double>(1,2)<<", 0, 0, 0, 1, 0]\n";
    r->success=true; r->message="saved " + output_file_ + ", reprojection error=" + std::to_string(err); RCLCPP_INFO(get_logger(), "%s", r->message.c_str());
  }
  const std::string preview_window_name_{"Captured calibration image"};
  std::string image_topic_, output_file_, captured_images_dir_; int board_width_, board_height_, required_frames_; double square_size_, capture_interval_; bool capturing_{false}, show_captured_image_{true}, preview_window_open_{false}; rclcpp::Time last_capture_; cv::Size image_size_; std::vector<std::vector<cv::Point2f>> image_points_; std::mutex mutex_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_; rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_srv_,stop_srv_,clear_srv_,calibrate_srv_;
};
int main(int argc,char** argv){rclcpp::init(argc,argv); rclcpp::spin(std::make_shared<CameraCalibrationNode>()); rclcpp::shutdown(); return 0;}
