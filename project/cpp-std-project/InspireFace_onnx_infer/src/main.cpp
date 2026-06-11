#include <iostream>
#include <opencv2/opencv.hpp>
#include "insight_face_onnx_detector.h"
#include <chrono>



int main() {
    std::string model_path = "/mnt/d/Data/code/Script/project/cpp-std-project/InspireFace_onnx_infer/resource/models/det_500m_with_pre.onnx";
    std::string image_path = "/mnt/d/Data/code/Script/project/cpp-std-project/InspireFace_onnx_infer/resource/face3.jpg";
    
    InsightFaceOnnxDetector model(model_path, 0.5, 0.4, 1, false, 0);  // true 表示启用 GPU + TensorRT
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to read image: " << image_path << std::endl;
        return -1;
    }

    std::vector<FaceDetectInfo> results = model.detect(img);
    for (const auto& face_info : results) {
        cv::rectangle(img, face_info.bbox, cv::Scalar(0, 255, 0), 2);
    }


    cv::imwrite("./scrfd_result1.jpg", img);
    std::cout << "[INFO] Result saved to: ./scrfd_result1.jpg" << std::endl;

}