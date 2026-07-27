#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "base_trt_infer.h"

struct FaceDetConfig
{
    std::string engine_path;
    float conf_threshold;
    float nms_threshold;
};

class FaceTrtDetector : public BaseTRTInfer
{

public:
    FaceTrtDetector(FaceDetConfig config) : m_config(config)
    {
        setOutputDim(2);
        // 加载模型
        loadEngine(m_config.engine_path);
        // 获取输入维度和类型
        hwcInput = m_inputDims.nbDims == 3 && m_inputDims.d[2] == 3;
        m_inputShape[0] = hwcInput ? m_inputDims.d[0] : m_inputDims.d[1];
        m_inputShape[1] = hwcInput ? m_inputDims.d[1] : m_inputDims.d[2];


    }

    // 进行推理
    std::vector<VisionCore::FaceDetection2D> infer(const cv::Mat &inputImage);
    // 获取配置
    FaceDetConfig getConfig() const { return m_config; }
    std::string getConfigStr() const;

private:
    FaceDetConfig m_config;
    cv::Mat m_resizedImage;
    const int m_fmc = 3; // 特征层数量
    const bool m_useKps = false;
    bool hwcInput = true;
    const std::vector<int> m_featStrides{8, 16, 32};
    std::vector<int> m_inputShape{640, 640};
    std::vector<VisionCore::FaceDetection2D> postprocessInsight(float scale, const cv::Size &originalSize) const;
    void processFeatureLayerInsight(size_t layerIdx,
                                    float scale,
                                    const cv::Size &originalSize,
                                    std::vector<float> &allScores,
                                    std::vector<float> &allBBoxes,
                                    std::vector<float> &allLandmarks) const;
    std::vector<VisionCore::FaceDetection2D> nms(const std::vector<VisionCore::FaceDetection2D> &detections) const;
};