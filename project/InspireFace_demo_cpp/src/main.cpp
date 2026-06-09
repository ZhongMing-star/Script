#include <iostream>
#include <string>
#include <inspireface.h>

#include "face_common.hpp"
#include "inspireface_wrapper.hpp"

int main()
{
    const std::string test_img_path = "test.jpg";
    const std::string model_pack_path = "/mnt/d/Data/code/Script/project/InspireFace_demo_cpp/source/models/Megatron_TRT";

    // 全局初始化 SDK
    try
    {
        InspireFaceGlobal::getInstance().Init(model_pack_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error initializing InspireFace SDK: " << e.what() << std::endl;
        return -1;
    }

    // 创建检测器和识别器
    InspireFaceDetector face_detector;
    InspireFaceFeatureExtract face_feature_extract;

    // 读取测试图像
    cv::Mat image = cv::imread(test_img_path);
    if (image.empty())
    {
        std::cerr << "Failed to load image: " << test_img_path << std::endl;
        return -1;
    }

    // 人脸检测
    std::vector<FaceDetectInfo> detect_info = face_detector.Detect(image);
    std::cout << "Detected " << detect_info.size() << " faces." << std::endl;

    // 人脸特征提取
    int i = 0;
    for (const auto &info : detect_info)
    {
        Face face = build_face(image, info);
        cv::imwrite("detected_face_" + std::to_string(i) + ".jpg", face.image); // 保存检测到的人脸图，供调试使用
        FaceFeature feature = face_feature_extract.Extract(face);
        std::cout << "Extracted feature of length: " << feature.size() << std::endl;
        i ++;
    }

    return 0;
}