#include "SCRFD.h"
#include <iostream>
#include <algorithm>
#include <cmath>

// SCRFD 构造函数：加载模型并初始化推理会话
SCRFD::SCRFD(const std::string& model_path, int intra_threads, bool use_gpu) {
    init_session(model_path, intra_threads, use_gpu);
}

// 初始化 ONNX Runtime 会话，支持 CPU 和 GPU 推理
void SCRFD::init_session(const std::string& model_path, int intra_threads, bool use_gpu) {
    session_options_.SetIntraOpNumThreads(intra_threads);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    // 如果启用 GPU，配置 CUDA provider 以加速推理
    if (use_gpu) {
        try {
            // 创建 CUDA provider 选项
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;  // 使用第一块 GPU 设备
            
            // 将 CUDA provider 添加到会话选项中
            // AppendExecutionProvider_CUDA 返回状态码，0 表示成功
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            
            std::cout << "[INFO] GPU (CUDA) provider enabled for inference" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] Failed to enable CUDA provider: " << e.what() 
                     << " - Falling back to CPU" << std::endl;
        }
    } else {
        std::cout << "[INFO] Using CPU for inference" << std::endl;
    }
    
    // 注：CPU provider 是 ONNX Runtime 的默认后备方案
    // 当 CUDA 不可用时，ONNX Runtime 会自动回退到 CPU 推理

    try {
        session_ = Ort::Session(env_, model_path.c_str(), session_options_);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Load model failed: " << e.what() << std::endl;
        throw;
    }

    size_t num_inputs = session_.GetInputCount();
    size_t num_outputs = session_.GetOutputCount();

    // 读取模型输入名和输入尺寸
    for (size_t i = 0; i < num_inputs; ++i) {
        // GetInputNameAllocated 返回一个托管指针（Ort::AllocatedStringPtr），
        // 该指针持有 ONNX Runtime 分配的输入名称内存
        // 使用 AllocatorWithDefaultOptions() 让 ONNX Runtime 自动管理内存的分配和释放
        auto name_ptr = session_.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        
        // 将输入名称的原始指针存储到 input_names_ 向量中
        // 这用于 session_.Run() 时作为输入绑定名称的数组
        input_names_.push_back(name_ptr.get());
        
        // 将托管指针本身移动到 input_names_ptr_ 向量中
        // 这确保在本对象销毁前，内存不被提前释放
        // 使用 std::move() 转移所有权，避免额外的拷贝开销
        input_names_ptr_.push_back(std::move(name_ptr));

        // 获取输入张量的类型和形状信息
        Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto dims = tensor_info.GetShape();

        // 验证输入是否为标准的 4 维张量 [N,C,H,W]（批次数、通道数、高、宽）
        // dims[0] = 1（批次数），dims[1] = 3（RGB 三通道）
        // dims[2] 和 dims[3] 分别是模型要求的输入高和宽
        if (dims.size() == 4 && dims[2] > 0 && dims[3] > 0) {
            // 从模型元数据中提取输入图像的宽度（dims[3]）和高度（dims[2]）
            // cv::Size 的参数顺序为 (宽, 高)
            input_size_ = cv::Size(static_cast<int>(dims[3]), static_cast<int>(dims[2]));
        } else {
            // 如果模型信息不完整或格式异常，使用默认的 640x640 输入尺寸
            input_size_ = cv::Size(640, 640);
            std::cout << "Input " << i << " default size: " << input_size_.width << "x" << input_size_.height << std::endl;
        }
    }

    // 读取模型输出名，用于推理时绑定输出
    // 类似输入名称的处理，分别存储原始指针和托管指针
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name_ptr = session_.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        output_names_.push_back(name_ptr.get());      // 用于 session_.Run() 的输出绑定
        output_names_ptr_.push_back(std::move(name_ptr));  // 保持内存生命周期
    }

    // 根据模型输出数量判断模型架构类型（有无关键点、特征层数等），并配置相应参数
    // fmc_: 特征金字塔的层数
    // feat_strides_: 各层相对于原图的步长
    // use_kps_: 是否使用关键点
    // num_anchors_: 每个网格点的锚框数量
    fmc_ = 3;
    feat_strides_ = {8, 16, 32};
    use_kps_ = false;
    num_anchors_ = 2;
}

// 预处理输入图像：缩放、填充、RGB 归一化、NCHW 排序
std::vector<float> SCRFD::preprocess(const cv::Mat& img, cv::Mat& resize_img, float& scale, float& dw, float& dh) {
    int img_h = img.rows;
    int img_w = img.cols;

    // 计算原图和模型输入的宽高比，用于确定最优的缩放方式
    float im_ratio = static_cast<float>(img_h) / img_w;      // 原图宽高比
    float model_ratio = static_cast<float>(input_size_.height) / input_size_.width;  // 模型输入宽高比

    int new_h, new_w;
    if (im_ratio > model_ratio) {
        // 原图相对更高，按高度缩放以填满模型的高度
        new_h = input_size_.height;
        new_w = static_cast<int>(new_h / im_ratio);
    } else {
        // 原图相对更宽，按宽度缩放以填满模型的宽度
        new_w = input_size_.width;
        new_h = static_cast<int>(new_w * im_ratio);
    }

    // 记录缩放因子，用于后续将模型输出坐标映射回原图
    scale = static_cast<float>(new_h) / img_h;

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_w, new_h));

    // 创建黑色背景（三通道 8 位无符号整数），然后将缩放后的图像居中填充
    // 这样可以保持原图长宽比，同时填满模型所需的输入尺寸
    resize_img = cv::Mat::zeros(input_size_.height, input_size_.width, CV_8UC3);
    // 计算水平和竖直方向的填充偏移量
    dw = (input_size_.width - new_w) / 2.0f;   // 水平填充距离（用于坐标逆变换）
    dh = (input_size_.height - new_h) / 2.0f;  // 竖直填充距离（用于坐标逆变换）
    // 将缩放后的图像复制到目标图像的中心区域
    resized_img.copyTo(resize_img(cv::Rect(static_cast<int>(dw), static_cast<int>(dh), new_w, new_h)));

    // 色彩空间转换：OpenCV 默认使用 BGR，但大多数深度学习模型使用 RGB
    cv::Mat rgb;
    cv::cvtColor(resize_img, rgb, cv::COLOR_BGR2RGB);
    // 转换为 32 位浮点数，便于数值运算
    rgb.convertTo(rgb, CV_32F);
    // 标准化处理：先减去均值（127.5），再除以标度（128.0）
    // 这样将像素值从 [0, 255] 映射到 [-1, 1] 范围
    rgb -= 127.5f;
    rgb /= 128.0f;

    // 计算输入张量所需的总元素数：1 个批次 × 3 个通道 × H × W
    size_t input_size = 1 * 3 * input_size_.height * input_size_.width;
    std::vector<float> input_tensor(input_size);
    int idx = 0;
    // ONNX Runtime 期望的数据排列为 NCHW 格式（批次、通道、高度、宽度）
    // 即对于 RGB 图像，先填充所有红色像素，再填充绿色像素，最后填充蓝色像素
    // OpenCV Mat 存储为 HWC 格式（高度、宽度、通道），需要重新排列
    for (int c = 0; c < 3; ++c) {        // 按通道迭代
        for (int h = 0; h < input_size_.height; ++h) {  // 按行迭代
            for (int w = 0; w < input_size_.width; ++w) {   // 按列迭代
                // 从 HWC 格式读取像素值，逐个添加到 NCHW 格式的输入张量
                input_tensor[idx++] = rgb.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return input_tensor;
}

// 从特征金字塔的单个层提取检测框、得分和关键点
// 该方法处理模型输出的一个特征层，提取该层中得分高于阈值的所有检测框
void SCRFD::processFeatureLayer(std::vector<Ort::Value>& outputs, size_t layer_idx,
                               float scale, float dw, float dh, float threshold, const cv::Size& original_size,
                               std::vector<float>& all_scores, std::vector<float>& all_bboxes,
                               std::vector<float>& all_landmarks) {
    int stride = feat_strides_[layer_idx];  // 该层相对于原图的下采样步长
    // 从 ONNX Runtime 输出张量中获取该层的检测得分
    float* scores = outputs[layer_idx].GetTensorMutableData<float>();
    // 获取该层的边界框回归值（相对于锚框的偏移量）
    float* bboxes = outputs[layer_idx + fmc_].GetTensorMutableData<float>();

    // 获取得分张量的形状，用于确定该层的锚框数量
    auto score_dims = outputs[layer_idx].GetTensorTypeAndShapeInfo().GetShape();
    int num_anchors = static_cast<int>(score_dims[0]);

    // 计算该层特征图的网格尺寸（输入图像尺寸 / 步长）
    int grid_h = input_size_.height / stride;  // 特征图的高度
    int grid_w = input_size_.width / stride;   // 特征图的宽度

    for (int i = 0; i < num_anchors; ++i) {
        float score = scores[i];  // 该锚框的检测置信度
        // 只保留高于阈值的检测框，过滤误检
        if (score < threshold) continue;

        // 根据锚框索引计算该锚框在特征图上的中心坐标
        // i/2 是网格位置的一维索引，转换为二维网格坐标
        // 再乘以步长，得到在输入图像坐标系中的位置
        float cx = ((i / 2) % grid_w) * stride;  // 中心 X 坐标
        float cy = ((i / 2) / grid_w) * stride;  // 中心 Y 坐标

        // 从回归输出中读取边界框的四个边界值（相对于锚框中心的距离）
        // SCRFD 的边界框表示方式：[左距离, 上距离, 右距离, 下距离]
        float l = bboxes[i * 4] * stride;         // 左边界距离中心的距离
        float t = bboxes[i * 4 + 1] * stride;     // 上边界距离中心的距离
        float r = bboxes[i * 4 + 2] * stride;     // 右边界距离中心的距离
        float b = bboxes[i * 4 + 3] * stride;     // 下边界距离中心的距离

        // 坐标变换：从模型输入图像坐标系映射回原图坐标系
        // 需要反向应用所有预处理变换：缩放、填充等
        // 根据锚框中心坐标和边界值计算边界框的四个角
        // (cx - l, cy - t) 是左上角，(cx + r, cy + b) 是右下角
        float x1 = (cx - l - dw) / scale;  // 左上角 X：先减去填充，再缩放回原图
        float y1 = (cy - t - dh) / scale;  // 左上角 Y
        float x2 = (cx + r - dw) / scale;  // 右下角 X
        float y2 = (cy + b - dh) / scale;  // 右下角 Y

        // 裁剪坐标到原图范围内，确保不超出边界
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_size.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_size.height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_size.width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_size.height)));

        all_scores.push_back(score);
        all_bboxes.insert(all_bboxes.end(), {x1, y1, x2, y2});

        if (use_kps_) {
            // 对于带关键点（landmarks）的模型，从对应输出中读取关键点坐标
            // 通常关键点用于表示人脸特征点（如眼睛、鼻子、嘴角等）
            float* landmarks = outputs[layer_idx + fmc_ * 2].GetTensorMutableData<float>();
            // 定位该锚框的关键点数据起始位置（每个锚框有 5 个关键点，每个 2 个坐标，共 10 个值）
            landmarks = landmarks + layer_idx * num_anchors * 10;

            // 迭代该锚框的 5 个关键点
            for (int k = 0; k < 5; ++k) {
                // 从模型输出中读取第 k 个关键点的 X、Y 坐标（相对于锚框中心的偏移）
                // 应用相同的坐标变换：缩放、去填充、映射回原图
                float lmx = (cx + landmarks[i * 10 + 2 * k] * stride - dw) / scale;
                float lmy = (cy + landmarks[i * 10 + 2 * k + 1] * stride - dh) / scale;
                all_landmarks.push_back(lmx);
                all_landmarks.push_back(lmy);
            }
        }
    }
}

// 对所有候选框执行非极大值抑制（NMS）
// 该方法根据 IoU 阈值过滤掉重叠过多的低分框，返回最终的检测结果
std::vector<Detection> SCRFD::performNMS(const std::vector<float>& all_scores, const std::vector<float>& all_bboxes,
                                         const std::vector<float>& all_landmarks, float nms_threshold) {
    std::vector<Detection> results;

    // 非极大值抑制（NMS）的准备工作：创建索引数组并按得分降序排序
    // 这样可以先处理高置信度的检测框，避免重复检测
    std::vector<int> indices(all_scores.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

    // 使用自定义比较函数按得分从高到低排序索引
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return all_scores[a] > all_scores[b];
    });

    // NMS 主循环：记录哪些检测框已被抑制
    std::vector<bool> suppressed(all_scores.size(), false);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        // 如果该框已被先前的高得分框抑制，则跳过
        if (suppressed[idx]) continue;

        // 构建检测结果结构体，存储该边界框及其属性
        Detection det;
        det.x1 = all_bboxes[idx * 4];      // 左上角 X
        det.y1 = all_bboxes[idx * 4 + 1];  // 左上角 Y
        det.x2 = all_bboxes[idx * 4 + 2];  // 右下角 X
        det.y2 = all_bboxes[idx * 4 + 3];  // 右下角 Y
        det.score = all_scores[idx];        // 检测得分

        // 如果有关键点信息，则将其存储到检测结果中
        if (use_kps_) {
            det.landmarks.resize(5);  // 分配 5 个关键点的空间
            for (int k = 0; k < 5; ++k) {
                // 每个关键点存储为 cv::Point2f (x, y)
                det.landmarks[k] = cv::Point2f(
                    all_landmarks[idx * 10 + 2 * k],      // 关键点 X 坐标
                    all_landmarks[idx * 10 + 2 * k + 1]   // 关键点 Y 坐标
                );
            }
        }

        // 将通过 NMS 的检测框添加到最终结果
        results.push_back(det);

        // 非极大值抑制（NMS）: 与当前框对比，抑制重叠过多的低分框
        // 只需对比后续未处理的框，因为前面的已经被处理过了
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;  // 如果已被抑制，跳过

            // 计算两个边界框的交集（Intersection）
            float xx1 = std::max(det.x1, all_bboxes[idx2 * 4]);          // 交集左上角 X
            float yy1 = std::max(det.y1, all_bboxes[idx2 * 4 + 1]);      // 交集左上角 Y
            float xx2 = std::min(det.x2, all_bboxes[idx2 * 4 + 2]);      // 交集右下角 X
            float yy2 = std::min(det.y2, all_bboxes[idx2 * 4 + 3]);      // 交集右下角 Y

            // 计算交集的面积（如果两个框不相交则为 0）
            float w = std::max(0.0f, xx2 - xx1);  // 交集宽度
            float h = std::max(0.0f, yy2 - yy1);  // 交集高度
            float inter = w * h;                  // 交集面积

            // 计算两个框的并集面积，用于计算 IoU（Intersection over Union）
            float area1 = (det.x2 - det.x1) * (det.y2 - det.y1);  // 框 1 的面积
            float area2 = (all_bboxes[idx2 * 4 + 2] - all_bboxes[idx2 * 4]) *
                         (all_bboxes[idx2 * 4 + 3] - all_bboxes[idx2 * 4 + 1]);  // 框 2 的面积
            float ovr = inter / (area1 + area2 - inter);  // IoU = 交集 / 并集

            // 如果 IoU 超过阈值，说明两个框重叠过多，应抑制低分框
            if (ovr >= nms_threshold) {
                suppressed[idx2] = true;
            }
        }
    }

    return results;
}

// 后处理模型输出，生成最终检测结果，包括 NMS 和关键点处理
std::vector<Detection> SCRFD::postprocess(std::vector<Ort::Value>& outputs, float scale, float dw, float dh,
                                           float threshold, float nms_threshold, const cv::Size& original_size) {
    std::vector<float> all_scores;
    std::vector<float> all_bboxes;
    std::vector<float> all_landmarks;

    // 特征金字塔处理：遍历多尺度特征层（步长分别为 8、16、32 等）
    // 从每个特征层中提取检测框、得分和关键点
    for (size_t idx = 0; idx < feat_strides_.size(); ++idx) {
        processFeatureLayer(outputs, idx, scale, dw, dh, threshold, original_size,
                           all_scores, all_bboxes, all_landmarks);
    }

    // 对所有候选框执行非极大值抑制，返回最终的检测结果
    return performNMS(all_scores, all_bboxes, all_landmarks, nms_threshold);
}

// 执行检测：输入图片 -> 预处理 -> 推理 -> 后处理 的完整流程
std::vector<Detection> SCRFD::detect(const cv::Mat& img, float threshold, float nms_threshold) {
    if (img.empty()) {
        std::cerr << "[ERROR] Empty input image" << std::endl;
        return {};
    }

    try {
        cv::Mat resize_img;
        float scale, dw, dh;
        // 第一步：预处理输入图像
        // 返回的 input_tensor 是 NCHW 格式的浮点数组
        // scale、dw、dh 用于后续将检测框坐标从模型输出映射回原图
        auto input_tensor = preprocess(img, resize_img, scale, dw, dh);

        // 第二步：准备 ONNX Runtime 推理所需的张量信息
        // 输入张量维度为 [1, 3, H, W]（批次大小、通道数、高、宽）
        std::vector<int64_t> input_dims = {1, 3, input_size_.height, input_size_.width};

        // 创建内存信息对象，指定使用 CPU 内存和竞技场分配器
        // 竞技场分配器能提供更好的性能（预分配大块内存）
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        // 创建输入张量：将预处理后的数据包装成 ONNX 张量格式
        // 注意：这里不拷贝数据，而是直接使用内存指针（需要确保数据有效期内调用 Run）
        auto input_mem = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor.data(),              // 浮点数据指针
            input_tensor.size(),              // 数据元素数量
            input_dims.data(),                // 张量维度数组
            input_dims.size()                 // 维度数
        );

        // 第三步：执行模型推理
        // session_.Run() 接收输入名、输入张量、输出名
        // 并返回 ONNX Runtime 分配的输出张量向量
        std::vector<Ort::Value> outputs = session_.Run(
            Ort::RunOptions{},           // 运行选项（使用默认设置）
            input_names_.data(),         // 输入名称数组
            &input_mem,                  // 输入张量数组
            input_names_.size(),         // 输入数量
            output_names_.data(),        // 输出名称数组
            output_names_.size()         // 输出数量
        );

        // 第四步：后处理模型输出，生成最终的检测结果
        return postprocess(outputs, scale, dw, dh, threshold, nms_threshold, img.size());
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Inference failed: " << e.what() << std::endl;
        return {};
    }
}
