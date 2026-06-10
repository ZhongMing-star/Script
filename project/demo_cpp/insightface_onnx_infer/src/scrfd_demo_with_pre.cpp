#include <iostream>
#include <opencv2/opencv.hpp>
#include "SCRFD_with_pre.h"
#include <chrono>

int main() {
    std::string model_path = "/mnt/d/Data/code/Script/project/demo_cpp/merge_preprocess_to_onnx/model/det_500m_with_pre.onnx";
    std::string image_path = "/mnt/d/Data/code/Script/project/demo_cpp/insightface_onnx_infer/resource/face2.jpg";
    int i;
    // 使用 TensorRT 进行推理：SCRFD(model_path, intra_threads, use_gpu=true, use_tensorrt=true)
    // 如果要使用 CPU 推理，将最后两个参数改为 false, false
    SCRFD_WITH_PRE detector(model_path, 1, false, false);  // true 表示启用 GPU + TensorRT
    std::cout << "[INFO] Model loaded successfully" << std::endl;
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to read image: " << image_path << std::endl;
        return -1;
    }
    
    for (i=0; i< 1; i++){
        try {
            
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Detection> results = detector.detect(img, 0.5f, 0.4f);
                auto end = std::chrono::high_resolution_clock::now();
                // 计算耗时：纳秒 / 毫秒 / 秒
            auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            double duration_s = duration_ns / 1000000000.0;

            std::cout << "耗时: " 
                    << duration_ns << " ns | "
                    << duration_ms << " ms | "
                    << duration_s << " s" << std::endl;
                    std::cout << "[INFO] Detected " << results.size() << " faces" << std::endl;
            
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

    }
    return 0;
}
