#pragma once

#include <opencv2/opencv.hpp>
#include <vector>


// ===================== 公共数据结构 =====================
// 单张人脸检测结果
/// @brief 人脸检测信息
/// @note 人脸检测信息包含了人脸框、检测置信度和 SDK 内部人脸标识
struct FaceDetectInfo
{
    cv::Rect bbox;   // 人脸框
    std::vector<cv::Point2f> landmarks;
    float det_score; // 检测置信度
};

/// @brief 人脸对象
/// @note 人脸对象包含了人脸框图、检测置信度和 SDK 内部人脸标识
struct Face
{
    // 人脸框图
    cv::Mat image;
    float det_score; // 检测置信度
};

/// @brief 根据原始图像和检测信息构建人脸对象
/// @param original_img 原始图像
/// @param detect_info 人脸检测信息
/// @return 人脸对象
inline Face build_face(const cv::Mat& original_img, const FaceDetectInfo& detect_info){
    return Face{original_img(detect_info.bbox).clone(), detect_info.det_score};
}

// 人脸特征向量
/// @brief 人脸特征向量
/// @note 人脸特征向量的维度根据 SDK 的实现而定
using FaceFeature = std::vector<double>;

// ===================== 人脸检测接口 =====================
// 人脸检测接口
class IFaceDetector
{
public:
    virtual ~IFaceDetector() = default;

    // 检测图像中的人脸
    virtual std::vector<FaceDetectInfo> detect(const cv::Mat& image) = 0;
};

// ===================== 人脸特征提取接口 =====================
// 人脸特征提取接口
class IFaceFeatureExtractor
{
public:
    virtual ~IFaceFeatureExtractor() = default;

    // 根据检测结果提取人脸特征
    virtual FaceFeature extract_feature(const Face& face) = 0;
};