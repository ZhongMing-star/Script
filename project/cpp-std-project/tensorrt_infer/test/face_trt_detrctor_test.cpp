#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "face_trt_detector.h"

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

/// 支持的图片扩展名列表
const std::vector<std::string> kImageExts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"};

bool isImageFile(const fs::path& path)
{
    if (!fs::is_regular_file(path)) {
        return false;
    }
    const std::string ext = path.extension().string();
    for (const auto& validExt : kImageExts) {
        if (ext == validExt) {
            return true;
        }
    }
    return false;
}

void printUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " <input_dir> [output_dir]" << std::endl;
    std::cout << "  input_dir    - 包含待推理图片的目录路径" << std::endl;
    std::cout << "  output_dir   - 结果保存目录（可选，默认在 input_dir 下创建 result 子目录）" << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    const fs::path inputDir = argv[1];
    if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
        std::cerr << "错误: 输入目录不存在或不是一个目录: " << inputDir << std::endl;
        return 1;
    }

    const fs::path outputDir = (argc >= 3) ? fs::path(argv[2]) : (inputDir / "result");
    fs::create_directories(outputDir);

    const fs::path defaultEnginePath = "/mnt/d/Data/code/ros2_ws/src/Vision/vision_core/resource/models/det_500m_end2end_int8_1280.trt";

    FaceDetConfig config;
    config.engine_path = getEnvOrDefault("FACE_TRT_ENGINE_PATH", defaultEnginePath.string());
    config.conf_threshold = std::stof(getEnvOrDefault("FACE_TRT_CONF_THRESHOLD", "0.25"));
    config.nms_threshold = std::stof(getEnvOrDefault("FACE_TRT_NMS_THRESHOLD", "0.25"));

    try {
        std::cout << "Loading engine: " << config.engine_path << std::endl;
        FaceTrtDetector detector(config);

        std::cout << detector.getConfigStr() << std::endl;

        // 收集目录下所有图片文件
        std::vector<fs::path> imageFiles;
        for (const auto& entry : fs::directory_iterator(inputDir)) {
            if (isImageFile(entry.path())) {
                imageFiles.push_back(entry.path());
            }
        }

        if (imageFiles.empty()) {
            std::cout << "在目录 " << inputDir << " 中未找到图片文件。" << std::endl;
            return 0;
        }

        std::cout << "找到 " << imageFiles.size() << " 张图片，开始推理..." << std::endl;

        for (size_t idx = 0; idx < imageFiles.size(); ++idx) {
            const fs::path& imgPath = imageFiles[idx];
            std::cout << "[" << (idx + 1) << "/" << imageFiles.size() << "] 处理: " << imgPath.filename() << std::endl;

            cv::Mat inputImage = cv::imread(imgPath.string());
            if (inputImage.empty()) {
                std::cerr << "  警告: 无法读取图片，跳过: " << imgPath << std::endl;
                continue;
            }

            const std::vector<VisionCore::FaceDetection2D> detections = detector.infer(inputImage);

            std::cout << "  检测到 " << detections.size() << " 个人脸" << std::endl;
            for (size_t i = 0; i < detections.size(); ++i) {
                const VisionCore::FaceDetection2D& detection = detections[i];
                std::cout << "    [" << i << "] cls=" << detection.cls_id
                          << ", conf=" << detection.conf
                          << ", bbox=(" << detection.bbox.x << ", " << detection.bbox.y
                          << ", " << detection.bbox.width << ", " << detection.bbox.height << ")"
                          << std::endl;
                cv::putText(inputImage, std::to_string(detection.cls_id) + " " + std::to_string(detection.conf),
                            cv::Point(static_cast<int>(detection.bbox.x), static_cast<int>(detection.bbox.y) - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                cv::rectangle(inputImage, detection.bbox, cv::Scalar(0, 255, 0), 2);
            }

            const fs::path outputPath = outputDir / imgPath.stem().concat("_result").concat(imgPath.extension().string());
            cv::imwrite(outputPath.string(), inputImage);
            std::cout << "  结果保存至: " << outputPath << std::endl;
        }

        std::cout << "所有图片处理完成！结果保存在: " << outputDir << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "face_trt_detector_test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}