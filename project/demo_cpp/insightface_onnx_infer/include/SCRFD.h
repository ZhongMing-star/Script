#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection {
    float x1, y1, x2, y2;
    float score;
    std::vector<cv::Point2f> landmarks;
};

class SCRFD {
public:
    // use_gpu: 是否使用 GPU 进行推理，默认为 true（支持 CUDA）
    SCRFD(const std::string& model_path, int intra_threads = 1, bool use_gpu = true);
    ~SCRFD() = default;

    std::vector<Detection> detect(const cv::Mat& img, float threshold = 0.5f, float nms_threshold = 0.4f);

private:
    void init_session(const std::string& model_path, int intra_threads, bool use_gpu);
    std::vector<float> preprocess(const cv::Mat& img, cv::Mat& resize_img, float& scale, float& dw, float& dh);
    std::vector<Detection> postprocess(std::vector<Ort::Value>& outputs, float scale, float dw, float dh, 
                                        float threshold, float nms_threshold, const cv::Size& original_size);
    
    // 从特征金字塔的单个层提取检测框、得分和关键点
    void processFeatureLayer(std::vector<Ort::Value>& outputs, size_t layer_idx, 
                            float scale, float dw, float dh, float threshold, const cv::Size& original_size,
                            std::vector<float>& all_scores, std::vector<float>& all_bboxes, 
                            std::vector<float>& all_landmarks);
    
    // 对所有候选框执行非极大值抑制（NMS）
    std::vector<Detection> performNMS(const std::vector<float>& all_scores, const std::vector<float>& all_bboxes,
                                      const std::vector<float>& all_landmarks, float nms_threshold);

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "SCRFD"};
    Ort::Session session_{nullptr};
    Ort::SessionOptions session_options_;

    std::vector<Ort::AllocatedStringPtr> input_names_ptr_;
    std::vector<Ort::AllocatedStringPtr> output_names_ptr_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    cv::Size input_size_;
    bool use_kps_ = false;
    int fmc_ = 3;
    std::vector<int> feat_strides_ = {8, 16, 32};
    int num_anchors_ = 1;
};
