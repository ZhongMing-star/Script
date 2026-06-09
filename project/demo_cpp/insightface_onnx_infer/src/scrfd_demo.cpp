#include <iostream>
#include <opencv2/opencv.hpp>
#include "SCRFD.h"

int main() {
    std::string model_path = "/mnt/d/Data/code/Script/project/demo_cpp/insightface_onnx_infer/resource/models/det_10g.onnx";
    std::string image_path = "/mnt/d/Data/code/Script/project/demo_cpp/insightface_onnx_infer/resource/face2.jpg";
    
    try {
        // 使用 GPU 进行推理：SCRFD(model_path, intra_threads, use_gpu=true)
        // 如果要使用 CPU 推理，将最后一个参数改为 false
        SCRFD detector(model_path, 1, true);  // true 表示启用 GPU (CUDA)
        std::cout << "[INFO] Model loaded successfully" << std::endl;
        
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cerr << "[ERROR] Failed to read image: " << image_path << std::endl;
            return -1;
        }
        
        std::vector<Detection> results = detector.detect(img, 0.5f, 0.4f);
        
        // std::cout << "[INFO] Detected " << results.size() << " faces" << std::endl;
        
        for (const auto& det : results) {
            if (det.score > 0.65){
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
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
