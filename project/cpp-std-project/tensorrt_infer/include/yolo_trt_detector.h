#pragma once 

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "base_trt_infer.h"

struct YoloDetConfig {
    std::string engine_path;
    int num_classes;
    float conf_threshold;
    float nms_threshold;
};


class YoloTrtDetector : public BaseTRTInfer {

public:
    YoloTrtDetector(YoloDetConfig config) : m_config(config) {
        // 加载模型
        loadEngine(m_config.engine_path);
    }

    // 进行推理
    std::vector<VisionCore::Detection2D> infer(const cv::Mat& inputImage);
    // 获取配置
    YoloDetConfig getConfig() const { return m_config; }
    std::string getConfigStr() const;

private:
    YoloDetConfig m_config;
    cv::Mat m_resizedImage;
    int featureCountFromDims(const nvinfer1::Dims& dims, int numClasses);
    std::vector<VisionCore::Detection2D> decode(const float* output, const VisionCore::LetterBoxInfo& letterboxInfo);
};