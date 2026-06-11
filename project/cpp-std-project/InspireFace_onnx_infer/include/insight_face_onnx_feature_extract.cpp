#include "insight_face_onnx_feature_extract.h"

#include <iostream>
#include <algorithm>
#include <cmath>


InspireFaceOnnxFeatureExtract::InspireFaceOnnxFeatureExtract(const std::string &model_path, int intra_threads, bool use_gpu, int device_id)
    :BaseOnnxInfer(model_path, intra_threads, use_gpu, device_id)
{

}

FaceFeature InspireFaceOnnxFeatureExtract::extract_feature(const Face& face)
{   
    if (face.image.empty())
    {
        std::cerr << "[ERROR] Empty input image" << std::endl;
        return {};
    }
    
    // 第一步： 前处理
    auto input_tensor = preprocess(face.image);

    // 第二步： 构造推理张量
    std::vector<int64_t> input_dims = {face.image.rows, face.image.cols, 3};
    // 创建内存信息对象，指定使用 CPU 内存和竞技场分配器
    // 竞技场分配器能提供更好的性能（预分配大块内存）
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    // 注意：这里不拷贝数据，而是直接使用内存指针（需要确保数据有效期内调用 Run）
    auto input_mem = Ort::Value::CreateTensor<uint8_t>(
        memory_info,
        input_tensor.data(), // 浮点数据指针
        input_tensor.size(), // 数据元素数量
        input_dims.data(),   // 张量维度数组
        input_dims.size()    // 维度数
    );

    // 第三步：执行模型推理
    // session_.Run() 接收输入名、输入张量、输出名
    // 并返回 ONNX Runtime 分配的输出张量向量
    std::vector<Ort::Value> outputs = m_session.Run(
        Ort::RunOptions{},    // 运行选项（使用默认设置）
        m_input_names.data(),  // 输入名称数组
        &input_mem,           // 输入张量数组
        m_input_names.size(),  // 输入数量
        m_output_names.data(), // 输出名称数组
        m_output_names.size()  // 输出数量
    );

    // 第四步：后处理模型输出，生成最终的检测结果
    return postprocess(outputs); // 后处理
}

std::vector<uint8_t> InspireFaceOnnxFeatureExtract::preprocess(const cv::Mat &img)
{
    int img_h = img.rows;
    int img_w = img.cols;

    // 计算输入张量所需的总元素数：H × W × 3
    // 此处耗时严重且占用 CPU 待优化
    size_t input_size = static_cast<size_t>(img_h) * img_w * 3;
    std::cout << "input_size = " << input_size << std::endl;
    std::vector<uint8_t> input_tensor(input_size);
    int idx = 0;
    for (int h = 0; h < img_h; ++h)
    {
        const cv::Vec3b *row = img.ptr<cv::Vec3b>(h);
        for (int w = 0; w < img_w; ++w)
        {
            for (int c = 0; c < 3; ++c)
            {
                input_tensor[idx++] = row[w][c];
            }
        }
    }
    return input_tensor;
}


FaceFeature InspireFaceOnnxFeatureExtract::postprocess(std::vector<Ort::Value> &outputs)
{
    int embedding_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    float *embeddings = outputs[0].GetTensorMutableData<float>();
    std::cout << "embedding_size = " << embedding_size << std::endl;
    FaceFeature feature;
    for (int i = 0; i < embedding_size; i ++)
    {
        feature.push_back(embeddings[i]);
    }
    return feature;
}

