#include <iostream>
#include <opencv2/opencv.hpp>
#include "fall_onnx_detectior.h"
#include <chrono>



int main() {
    std::string model_path = "/mnt/d/Data/code/Script/project/cpp-std-project/InspireFace_onnx_infer/resource/models/fall_detection_fp16_with_pre_640.onnx";
    std::string image_path = "/mnt/d/Data/code/Script/project/cpp-std-project/InspireFace_onnx_infer/resource/fall3.jpg";
    
    FallOnnxDetector model(model_path, 0.3, 0.1, 1, false, 0);  // true 表示启用 GPU + TensorRT

    std::map<std::string , std::string> label_info = {
        {"0", "Normal"},
        {"1", "Fall"},
    };

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to read image: " << image_path << std::endl;
        return -1;
    }
    std::vector<DetectInfo> results = model.detect(img);
    std::cout << "Number of detections: " << results.size() << std::endl;
    // 绘制结果并保存
    for(auto res : results){
        cv::rectangle(img, cv::Point(res.bbox.x, res.bbox.y), cv::Point(res.bbox.x + res.bbox.width, res.bbox.y + res.bbox.height), cv::Scalar(0, 255, 0), 2);
        // 绘制类别
        
        std::cout << "res.id: " << res.id << " \t res.score: " << res.score << std::endl;
        cv::putText(img, label_info[std::to_string(res.id)], cv::Point(res.bbox.x, res.bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("result.jpg", img);
    return 0;

}