#pragma once

#include <inspireface.h>
#include <herror.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

#include "face_common.hpp"

class InspireFaceGlobal
{
public:
    // 禁止拷贝构造和赋值
    InspireFaceGlobal(const InspireFaceGlobal &) = delete;
    InspireFaceGlobal &operator=(const InspireFaceGlobal &) = delete;
    InspireFaceGlobal(InspireFaceGlobal &&) = delete;
    InspireFaceGlobal &operator=(InspireFaceGlobal &&) = delete;

    // 获取单例
    static InspireFaceGlobal &getInstance()
    {
        static InspireFaceGlobal instance;
        return instance;
    }

    // 全局初始化 SDK
    bool Init(const std::string &model_pack_path)
    {
        if (m_inited)
            return true; // 已经初始化过了

        // 初始化 SDK
        HResult ret = HFReloadInspireFace(model_pack_path.c_str());
        if (ret != HSUCCEED)
        {
            throw std::runtime_error("Failed to launch InspireFace: " + std::to_string(ret));
        }

        m_model_path = model_pack_path;
        m_inited = true;
        return true;
    }

    ~InspireFaceGlobal()
    {
        if (m_inited)
        {
            HFTerminateInspireFace();
            m_inited = false;
        }
    }

    bool IsInited() const { return m_inited; }
    std::string GetModelPath() const { return m_model_path; }

private:
    InspireFaceGlobal() : m_inited(false) {};
    bool m_inited = false;
    std::string m_model_path;
};

class InspireFaceDetector : public IFaceDetector
{
public:
    InspireFaceDetector(int max_faces_num = 10)
    {
        if (!InspireFaceGlobal::getInstance().IsInited())
        {
            throw std::runtime_error("InspireFace SDK is not initialized. Please call InspireFaceGlobal::getInstance().Init() first.");
        }

        HOption options = HF_ENABLE_QUALITY;
        HFDetectMode model = HF_DETECT_MODE_ALWAYS_DETECT;
        HResult ret = HFCreateInspireFaceSessionOptional(options, model, max_faces_num, -1, 0, &m_session);
        if (ret != HSUCCEED)
        {
            throw std::runtime_error("Failed to create InspireFace session: " + std::to_string(ret));
        }
    }

    ~InspireFaceDetector()
    {
        if (m_session != nullptr)
        {
            HFReleaseInspireFaceSession(m_session);
            m_session = nullptr;
        }
    }

    InspireFaceDetector(const InspireFaceDetector &) = delete;
    InspireFaceDetector &operator=(const InspireFaceDetector &) = delete;

    std::vector<FaceDetectInfo> Detect(const cv::Mat &image)
    {
        std::vector<FaceDetectInfo> results;
        if (image.empty() || m_session == nullptr)
            return results;

        HFImageBitmap bitmap = nullptr;
        HFImageStream stream = nullptr;

        // 1. 按 SDK 定义的方式，构造 HImageBitmapData 结构体
        HFImageBitmapData bitmapData;
        bitmapData.data = image.data;
        bitmapData.width = image.cols;
        bitmapData.height = image.rows;
        bitmapData.channels = image.channels();

        // 2. 创建图像流
        HResult ret = HFCreateImageBitmap(&bitmapData, &bitmap);
        if (ret != HSUCCEED)
        {
            throw std::runtime_error("Failed to create image stream: " + std::to_string(ret));
        }

        // 3. 从 Bitmap 创建 ImageStream
        ret = HFCreateImageStreamFromImageBitmap(bitmap, HF_CAMERA_ROTATION_0, &stream);
        if (ret != HSUCCEED)
        {
            HFReleaseImageBitmap(bitmap);
            throw std::runtime_error("Failed to create image stream from bitmap: " + std::to_string(ret));
        }

        // 4. 执行人脸检测
        HFMultipleFaceData faces;
        ret = HFExecuteFaceTrack(m_session, stream, &faces);
        if (ret != HSUCCEED)
        {
            HFReleaseImageBitmap(bitmap);
            HFReleaseImageStream(stream);
            throw std::runtime_error("Failed to execute face track: " + std::to_string(ret));
        }

        // 5. 解析检测结果
        for (int i = 0; i < faces.detectedNum; ++i)
        {
            FaceDetectInfo info;
            info.bbox = cv::Rect(
                faces.rects[i].x, faces.rects[i].y,
                faces.rects[i].width, faces.rects[i].height);

            info.det_score = faces.detConfidence[i];
            info.token = faces.tokens[i];
            results.push_back(info);
        }

        // 6. 释放资源
        HFReleaseImageBitmap(bitmap);
        HFReleaseImageStream(stream);

        return results;
    }

private:
    HFSession m_session;
};

class InspireFaceFeatureExtract : public IFaceFeatureExtractor
{
public:
    /// @brief 人脸识别器
    /// @param max_faces_num 最大人脸检测数量
    InspireFaceFeatureExtract(int max_faces_num = 10)
    {
        if (!InspireFaceGlobal::getInstance().IsInited())
        {
            throw std::runtime_error("InspireFace SDK is not initialized. Please call InspireFaceGlobal::getInstance().Init() first.");
        }

        HOption options = HF_ENABLE_FACE_RECOGNITION;
        HFDetectMode model = HF_DETECT_MODE_ALWAYS_DETECT;
        HResult ret = HFCreateInspireFaceSessionOptional(options, model, max_faces_num, -1, 0, &m_session);
        if (ret != HSUCCEED)
        {
            throw std::runtime_error("Failed to create InspireFace session: " + std::to_string(ret));
        }
    }

    /// @brief 人脸识别器析构函数
    ~InspireFaceFeatureExtract()
    {
        if (m_session != nullptr)
        {
            HFReleaseInspireFaceSession(m_session);
            m_session = nullptr;
        }
    }

    // 禁用复制构造函数和赋值运算符
    InspireFaceFeatureExtract(const InspireFaceFeatureExtract &) = delete;
    InspireFaceFeatureExtract &operator=(const InspireFaceFeatureExtract &) = delete;

    /// @brief 人脸识别器识别函数
    /// @param img 输入图像
    /// @param detect_info 人脸检测信息
    /// @return 人脸特征
    FaceFeature Extract(const Face &face)
    {
        FaceFeature face_feature;
        if (face.image.empty() || m_session == nullptr)
            return face_feature;
        
        HFImageBitmap bitmap = nullptr;
        HFImageStream stream = nullptr;
        HFFaceFeature feature;

        // 1. 按 SDK 定义的方式，构造 HImageBitmapData 结构体
        HFImageBitmapData bitmapData;
        bitmapData.data = face.image.data;
        bitmapData.width = face.image.cols;
        bitmapData.height = face.image.rows;
        bitmapData.channels = face.image.channels();

        // 2. 创建图像流
        HResult ret = HFCreateImageBitmap(&bitmapData, &bitmap);
        if (ret != HSUCCEED)
        {
            throw std::runtime_error("Failed to create image stream: " + std::to_string(ret));
        }

        // 3. 从 Bitmap 创建 ImageStream
        ret = HFCreateImageStreamFromImageBitmap(bitmap, HF_CAMERA_ROTATION_0, &stream);
        if (ret != HSUCCEED)
        {
            HFReleaseImageBitmap(bitmap);
            throw std::runtime_error("Failed to create image stream from bitmap: " + std::to_string(ret));
        }

        // 4. 执行人脸识别
        ret = HFFaceFeatureExtract(m_session, stream, face.token, &feature);
        if (ret != HSUCCEED)
        {
            HFReleaseImageBitmap(bitmap);
            HFReleaseImageStream(stream);
            throw std::runtime_error("Failed to execute face recognition: " + std::to_string(ret));
        }

        // 5. 转换为 vector 输出
        face_feature.assign(feature.data, feature.data + feature.size);

        // 6. 释放资源
        HFReleaseImageBitmap(bitmap);
        HFReleaseImageStream(stream);

        return face_feature;
    }

private:
    HFSession m_session;
};