#include <iostream>
#include <opencv2/opencv.hpp>
#include "insight_face_onnx_infer.h"
#include <chrono>



int main() {
    std::string model_path = "/mnt/d/Data/code/Script/project/cpp-std-project/InspireFace_onnx_infer/resource/models/det_500m_with_pre.onnx";
    std::string image_path = "/mnt/d/Data/code/Script/project/cpp-std-project/InspireFace_onnx_infer/resource/face3.jpg";
    
    InsightFaceOnnxInfer detector(model_path, 0.5, 0.4, 1, false, 0);  // true 表示启用 GPU + TensorRT
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to read image: " << image_path << std::endl;
        return -1;
    }

    std::vector<Detection> results = detector.detect(img);
    std::cout << "results size = " << results.size() << std::endl;
    for (const auto& det : results) {
        if (det.score > 0.5){
            cv::rectangle(img, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), cv::Scalar(0, 255, 0), 2);
        }
        if (!det.landmarks.empty() && det.score > 0.65) {
            for (const auto& kp : det.landmarks) {
                cv::circle(img, kp, 3, cv::Scalar(0, 0, 255), -1);
            }
        }
    }
    
    cv::imwrite("./scrfd_result1.jpg", img);
    std::cout << "[INFO] Result saved to: ./scrfd_result1.jpg" << std::endl;

}