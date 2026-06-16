#pragma once 

#include "base_onnx_infer.h"
#include <opencv2/opencv.hpp>

struct DetectInfo
{
    cv::Rect bbox;
    float score;
    int id;
};


class FallOnnxDetector: public BaseOnnxInfer
{
public:
    FallOnnxDetector(const std::string &model_path, float threshold, float nms_threshold, int intra_threads, bool use_gpu, int device_id);
    ~FallOnnxDetector() = default;

    std::vector<DetectInfo> detect(const cv::Mat& image);
    
private:
    float m_threshold;
    float m_nms_threshold;
    const cv::Size m_input_size = cv::Size(640, 640);
    
    void preprocess(const cv::Mat& img, float& scale);
    std::vector<DetectInfo> postprocess(std::vector<Ort::Value>& output, float& scale, cv::Size input_size);
    void nms(std::vector<DetectInfo>& detect_info);
};