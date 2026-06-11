#include <iostream>
#include <opencv2/opencv.hpp>
#include "insight_face_onnx_feature_extract.h"
#include <chrono>



int main() {
    std::string model_path = "/mnt/d/Data/code/Script/project/cpp-std-project/InspireFace_onnx_infer/resource/models/w600k_mbf_with_with_pre.onnx";
    std::string image_path = "/mnt/d/Data/code/Script/project/cpp-std-project/InspireFace_onnx_infer/resource/face3.jpg";
    
    InspireFaceOnnxFeatureExtract model(model_path, 1, false, 0);  // true 表示启用 GPU + TensorRT
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to read image: " << image_path << std::endl;
        return -1;
    }
    Face face{img, 0};
    FaceFeature results = model.extract_feature(face);
    std::cout << "results size = " << results.size() << std::endl;

}