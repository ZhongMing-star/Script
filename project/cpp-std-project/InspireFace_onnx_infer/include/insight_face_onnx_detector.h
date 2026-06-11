#pragma once 

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

#include "base_onnx_infer.h"
#include "face_common.hpp"

class InsightFaceOnnxDetector : public BaseOnnxInfer, IFaceDetector
{
public:

    /**
     * @brief 构造函数，用于初始化基于ONNX运行时的InsightFace推理器
     * 
     * @param model_path      模型文件的路径（如.onnx文件的路径）
     * @param threshold       人脸检测或识别的置信度阈值，默认值为0.5
     * @param nms_threshold   非极大值抑制（NMS）的阈值，用于过滤重叠的边界框，默认值为0.4
     * @param intra_threads   ONNX运行时算子内部的线程数，用于控制并发执行，默认值为1
     * @param use_gpu         是否使用GPU进行推理加速，默认值为true
     */
    InsightFaceOnnxDetector(const std::string& model_path, float threshold = 0.5, float nms_threshold = 0.4, int intra_threads = 1, bool use_gpu = true, int device_id = 0);


    /**
     * @brief 析构函数：销毁 InsightFaceOnnxDetector 对象，释放相关资源。
     * @note 使用 = default 显式指定编译器生成默认析构函数。
     *       表明该类无需自定义析构逻辑，编译器自动生成的默认实现足以正确清理对象。
     */
    ~InsightFaceOnnxDetector() = default; // 默认析构函数，无需手动释放资源

    /**
     * @brief 人脸检测函数，用于在输入图像中检测人脸。
     * 
     * @param img 输入图像，类型为cv::Mat
     * @return std::vector<FaceDetectInfo> 检测到的人脸检测结果，每个检测结果包含边界框坐标、置信度和关键点坐标
     */
    std::vector<FaceDetectInfo> detect(const cv::Mat &img);

private:
    float m_threshold;
    float m_nms_threshold;


    // 模型的输入就是 640 * 640
    const cv::Size m_input_size = cv::Size(640, 640);
    const int m_fmc = 3;
    const int m_num_anchors = 1;
    const bool m_use_kps = false;
    const std::vector<int> m_feat_strides = {8, 16, 32};

    // 前处理， 计算图像 resize 相关参数
    std::vector<uint8_t> preprocess(const cv::Mat &img, float &scale);
    // 后处理
    // 对模型输出进行后处理
    std::vector<FaceDetectInfo> postprocess(std::vector<Ort::Value> &outputs, float scale, 
                                          const cv::Size &original_size);

    // 从特征金字塔的单个层提取检测框、得分和关键点
    void processFeatureLayer(std::vector<Ort::Value>& outputs, size_t layer_idx, 
                            float scale, float threshold, const cv::Size& input_size, const cv::Size& original_size,
                            std::vector<float>& all_scores, std::vector<float>& all_bboxes, 
                            std::vector<float>& all_landmarks);

    // 对所有候选框执行非极大值抑制（NMS）
    std::vector<FaceDetectInfo> performNMS(const std::vector<float>& all_scores, const std::vector<float>& all_bboxes,
                                    const std::vector<float>& all_landmarks, float nms_threshold);

};