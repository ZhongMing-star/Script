#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>

#include "yolo_trt_detector.h"

namespace fs = std::filesystem;

namespace {

std::string getEnvOrDefault(const char* name, const std::string& defaultValue)
{
    const char* value = std::getenv(name);
    if (value != nullptr && value[0] != '\0') {
        return value;
    }
    return defaultValue;
}

}  // namespace

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    const fs::path sourceDir = fs::path(VISION_CORE_SOURCE_DIR);
    const fs::path defaultEnginePath = sourceDir / "resource" / "models" / "fall_detection_end2end_fp16_640.trt";

    YoloDetConfig config;
    config.engine_path = getEnvOrDefault("YOLO_TRT_ENGINE_PATH", defaultEnginePath.string());
    config.num_classes = std::stoi(getEnvOrDefault("YOLO_TRT_NUM_CLASSES", "1"));
    config.conf_threshold = std::stof(getEnvOrDefault("YOLO_TRT_CONF_THRESHOLD", "0.25"));
    config.nms_threshold = std::stof(getEnvOrDefault("YOLO_TRT_NMS_THRESHOLD", "0.25"));

    try {
        std::cout << "Loading engine: " << config.engine_path << std::endl;
        YoloTrtDetector detector(config);

        std::cout << detector.getConfigStr() << std::endl;

        cv::Mat inputImage = cv::imread("/mnt/d/Data/code/Script/project/cpp-std-project/tensorrt_infer/resource/images/split8_199.png");
        const std::vector<VisionCore::Detection2D> detections = detector.infer(inputImage);

        std::cout << "Detection count: " << detections.size() << std::endl;
        for (size_t i = 0; i < detections.size(); ++i) {
            const VisionCore::Detection2D& detection = detections[i];
            std::cout << "[" << i << "] cls=" << detection.cls_id
                      << ", conf=" << detection.conf
                      << ", bbox=(" << detection.bbox.x << ", " << detection.bbox.y
                      << ", " << detection.bbox.width << ", " << detection.bbox.height << ")"
                      << std::endl;
            cv::putText(inputImage, std::to_string(detection.cls_id) + " " + std::to_string(detection.conf),
                        cv::Point(static_cast<int>(detection.bbox.x), static_cast<int>(detection.bbox.y) - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            cv::rectangle(inputImage, detection.bbox, cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite("/mnt/d/Data/code/Script/project/cpp-std-project/tensorrt_infer/resource/test_img/face2_result.jpg", inputImage);
        std::cout << "Result image saved to: /mnt/d/Data/code/Script/project/cpp-std-project/tensorrt_infer/resource/test_img/face2_result.jpg" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "yolo_trt_detector_smoke_test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}