#include "insight_face_onnx_infer.h"

#include <iostream>
#include <algorithm>
#include <cmath>

InsightFaceOnnxInfer::InsightFaceOnnxInfer(const std::string &model_path, float threshold, float nms_threshold, int intra_threads, bool use_gpu, int device_id)
{
    // 初始化成员变量
    m_model_path = model_path;
    m_threshold = threshold;
    m_nms_threshold = nms_threshold;
    m_intra_threads = intra_threads;
    m_use_gpu = use_gpu;
    m_deivce_id = device_id;

    // 初始化 session
    init_session();
    // 初始化模型
    init_model();
}

void InsightFaceOnnxInfer::init_session()
{
    m_session_options.SetIntraOpNumThreads(m_intra_threads);
    m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    if (m_use_gpu)
    {
        try
        {
            // 创建 CUDA provider 选项
            OrtCUDAProviderOptions cuda_options;
            // 使用第一块 GPU 设备
            cuda_options.device_id = m_deivce_id;

            // 将 CUDA provider 添加到会话选项中
            // AppendExecutionProvider_CUDA 返回状态码，0 表示成功
            m_session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[INFO] GPU (CUDA) provider enabled for inference" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "CUDA initialization failed: " << e.what() << std::endl;
            m_use_gpu = false;
        }
    }

    // 注：CPU provider 是 ONNX Runtime 的默认后备方案
    // 当 CUDA 不可用时，ONNX Runtime 会自动回退到 CPU 推理
    try
    {
        m_session = Ort::Session(m_env, m_model_path.c_str(), m_session_options);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] Load model failed: " << e.what() << std::endl;
        throw;
    }
}

void InsightFaceOnnxInfer::init_model()
{
    size_t num_inputs = m_session.GetInputCount();
    size_t num_outputs = m_session.GetOutputCount();

    // 读取模型输入名和输入尺寸
    for (size_t i = 0; i < num_inputs; ++i)
    {
        // GetInputNameAllocated 返回一个托管指针（Ort::AllocatedStringPtr），
        // 该指针持有 ONNX Runtime 分配的输入名称内存
        // 使用 AllocatorWithDefaultOptions() 让 ONNX Runtime 自动管理内存的分配和释放
        auto name_ptr = m_session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());

        // 将输入名称的原始指针存储到 m_input_names 向量中
        // 这用于 session_.Run() 时作为输入绑定名称的数组
        m_input_names.push_back(name_ptr.get());

        // 将托管指针本身移动到 m_input_names_ptr 向量中
        // 这确保在本对象销毁前，内存不被提前释放
        // 使用 std::move() 转移所有权，避免额外的拷贝开销
        m_input_names_ptr.push_back(std::move(name_ptr));
    }

    // 读取模型输出名，用于推理时绑定输出
    // 类似输入名称的处理，分别存储原始指针和托管指针
    for (size_t i = 0; i < num_outputs; ++i)
    {
        auto name_ptr = m_session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        // 用于 session_.Run() 的输出绑定
        m_output_names.push_back(name_ptr.get());
        // 保持内存生命周期
        m_output_names_pt_.push_back(std::move(name_ptr));
    }
}


std::vector<Detection> InsightFaceOnnxInfer::detect(cv::Mat &img)
{
    if (img.empty())
    {
        std::cerr << "[ERROR] Empty input image" << std::endl;
        return {};
    }
    
    float scale;
    // 第一步： 前处理
    auto input_tensor = preprocess(img, scale);

    // 第二步： 构造推理张量
    std::vector<int64_t> input_dims = {img.rows, img.cols, 3};
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
    auto res = postprocess(outputs, scale, img.size()); // 后处理
    
    return res;
}

std::vector<uint8_t> InsightFaceOnnxInfer::preprocess(const cv::Mat &img, float &scale)
{
    int img_h = img.rows;
    int img_w = img.cols;

    // 计算 resize 比例
    // 计算原图和模型输入的宽高比，用于后续坐标逆变换
    float im_ratio = static_cast<float>(img_h) / img_w;
    float model_ratio = static_cast<float>(m_input_size.height) / m_input_size.width;

    int new_h, new_w;
    if (im_ratio > model_ratio)
    {
        new_h = m_input_size.height;
        new_w = static_cast<int>(new_h / im_ratio);
    }
    else
    {
        new_w = m_input_size.width;
        new_h = static_cast<int>(new_w * im_ratio);
    }

    // 内部预处理会在模型里执行 Resize/Pad，因此这里只记录映射关系
    scale = static_cast<float>(new_h) / img_h;

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


std::vector<Detection> InsightFaceOnnxInfer::postprocess(std::vector<Ort::Value> &outputs, float scale, 
                                          const cv::Size &original_size)
{
    std::vector<float> all_scores;
    std::vector<float> all_bboxes;
    std::vector<float> all_landmarks;

    // 特征金字塔处理：遍历多尺度特征层（步长分别为 8、16、32 等）
    // 从每个特征层中提取检测框、得分和关键点
    for (size_t idx = 0; idx < m_feat_strides.size(); ++idx)
    {
        processFeatureLayer(outputs, idx, scale, m_threshold, m_input_size, original_size,
                            all_scores, all_bboxes, all_landmarks);
    }
    std::cout << "all_bboxes.size() = " << all_bboxes.size() << std::endl;

    return performNMS(all_scores, all_bboxes, all_landmarks, m_nms_threshold);
    
}


// 从特征金字塔的单个层提取检测框、得分和关键点
// 该方法处理模型输出的一个特征层，提取该层中得分高于阈值的所有检测框
void InsightFaceOnnxInfer::processFeatureLayer(std::vector<Ort::Value> &outputs, size_t layer_idx,
                                float scale, float threshold, const cv::Size &input_size, const cv::Size &original_size,
                                std::vector<float> &all_scores, std::vector<float> &all_bboxes,
                                std::vector<float> &all_landmarks)
{
    int stride = m_feat_strides[layer_idx]; // 该层相对于原图的下采样步长
    // 从 ONNX Runtime 输出张量中获取该层的检测得分
    float *scores = outputs[layer_idx].GetTensorMutableData<float>();
    // 获取该层的边界框回归值（相对于锚框的偏移量）
    float *bboxes = outputs[layer_idx + m_fmc].GetTensorMutableData<float>();

    // 获取得分张量的形状，用于确定该层的锚框数量
    auto score_dims = outputs[layer_idx].GetTensorTypeAndShapeInfo().GetShape();
    int num_anchors = static_cast<int>(score_dims[0]);

    // 计算该层特征图的网格尺寸（输入图像尺寸 / 步长）
    int grid_h = input_size.height / stride; // 特征图的高度
    int grid_w = input_size.width / stride;  // 特征图的宽度

    for (int i = 0; i < num_anchors; ++i)
    {
        float score = scores[i]; // 该锚框的检测置信度
        // 只保留高于阈值的检测框，过滤误检
        if (score < threshold)
            continue;

        // 根据锚框索引计算该锚框在特征图上的中心坐标
        // i/2 是网格位置的一维索引，转换为二维网格坐标
        // 再乘以步长，得到在输入图像坐标系中的位置
        float cx = ((i / 2) % grid_w) * stride; // 中心 X 坐标
        float cy = ((i / 2) / grid_w) * stride; // 中心 Y 坐标

        // 从回归输出中读取边界框的四个边界值（相对于锚框中心的距离）
        // SCRFD_WITH_PRE 的边界框表示方式：[左距离, 上距离, 右距离, 下距离]
        float l = bboxes[i * 4] * stride;     // 左边界距离中心的距离
        float t = bboxes[i * 4 + 1] * stride; // 上边界距离中心的距离
        float r = bboxes[i * 4 + 2] * stride; // 右边界距离中心的距离
        float b = bboxes[i * 4 + 3] * stride; // 下边界距离中心的距离

        // 坐标变换：从模型输入图像坐标系映射回原图坐标系
        // 需要反向应用所有预处理变换：缩放、填充等
        // 根据锚框中心坐标和边界值计算边界框的四个角
        // (cx - l, cy - t) 是左上角，(cx + r, cy + b) 是右下角
        float x1 = (cx - l ) / scale; // 左上角 X：先减去填充，再缩放回原图
        float y1 = (cy - t ) / scale; // 左上角 Y
        float x2 = (cx + r) / scale; // 右下角 X
        float y2 = (cy + b) / scale; // 右下角 Y

        // 裁剪坐标到原图范围内，确保不超出边界
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_size.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_size.height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_size.width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_size.height)));

        all_scores.push_back(score);
        all_bboxes.insert(all_bboxes.end(), {x1, y1, x2, y2});

        if (m_use_kps)
        {
            // 对于带关键点（landmarks）的模型，从对应输出中读取关键点坐标
            // 通常关键点用于表示人脸特征点（如眼睛、鼻子、嘴角等）
            float *landmarks = outputs[layer_idx + m_fmc * 2].GetTensorMutableData<float>();
            // 定位该锚框的关键点数据起始位置（每个锚框有 5 个关键点，每个 2 个坐标，共 10 个值）
            landmarks = landmarks + layer_idx * num_anchors * 10;

            // 迭代该锚框的 5 个关键点
            for (int k = 0; k < 5; ++k)
            {
                // 从模型输出中读取第 k 个关键点的 X、Y 坐标（相对于锚框中心的偏移）
                // 应用相同的坐标变换：缩放、去填充、映射回原图
                float lmx = (cx + landmarks[i * 10 + 2 * k] * stride) / scale;
                float lmy = (cy + landmarks[i * 10 + 2 * k + 1] * stride) / scale;
                all_landmarks.push_back(lmx);
                all_landmarks.push_back(lmy);
            }
        }
    }
}

// 对所有候选框执行非极大值抑制（NMS）
// 该方法根据 IoU 阈值过滤掉重叠过多的低分框，返回最终的检测结果
std::vector<Detection> InsightFaceOnnxInfer::performNMS(const std::vector<float> &all_scores, const std::vector<float> &all_bboxes,
                                         const std::vector<float> &all_landmarks, float nms_threshold)
{
    std::vector<Detection> results;

    // 非极大值抑制（NMS）的准备工作：创建索引数组并按得分降序排序
    // 这样可以先处理高置信度的检测框，避免重复检测
    std::vector<int> indices(all_scores.size());
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;

    // 使用自定义比较函数按得分从高到低排序索引
    std::sort(indices.begin(), indices.end(), [&](int a, int b)
              { return all_scores[a] > all_scores[b]; });

    // NMS 主循环：记录哪些检测框已被抑制
    std::vector<bool> suppressed(all_scores.size(), false);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        // 如果该框已被先前的高得分框抑制，则跳过
        if (suppressed[idx])
            continue;

        // 构建检测结果结构体，存储该边界框及其属性
        Detection det;
        det.x1 = all_bboxes[idx * 4];     // 左上角 X
        det.y1 = all_bboxes[idx * 4 + 1]; // 左上角 Y
        det.x2 = all_bboxes[idx * 4 + 2]; // 右下角 X
        det.y2 = all_bboxes[idx * 4 + 3]; // 右下角 Y
        det.score = all_scores[idx];      // 检测得分

        // 如果有关键点信息，则将其存储到检测结果中
        if (m_use_kps)
        {
            det.landmarks.resize(5); // 分配 5 个关键点的空间
            for (int k = 0; k < 5; ++k)
            {
                // 每个关键点存储为 cv::Point2f (x, y)
                det.landmarks[k] = cv::Point2f(
                    all_landmarks[idx * 10 + 2 * k],    // 关键点 X 坐标
                    all_landmarks[idx * 10 + 2 * k + 1] // 关键点 Y 坐标
                );
            }
        }

        // 将通过 NMS 的检测框添加到最终结果
        results.push_back(det);

        // 非极大值抑制（NMS）: 与当前框对比，抑制重叠过多的低分框
        // 只需对比后续未处理的框，因为前面的已经被处理过了
        for (size_t j = i + 1; j < indices.size(); ++j)
        {
            int idx2 = indices[j];
            if (suppressed[idx2])
                continue; // 如果已被抑制，跳过

            // 计算两个边界框的交集（Intersection）
            float xx1 = std::max(det.x1, all_bboxes[idx2 * 4]);     // 交集左上角 X
            float yy1 = std::max(det.y1, all_bboxes[idx2 * 4 + 1]); // 交集左上角 Y
            float xx2 = std::min(det.x2, all_bboxes[idx2 * 4 + 2]); // 交集右下角 X
            float yy2 = std::min(det.y2, all_bboxes[idx2 * 4 + 3]); // 交集右下角 Y

            // 计算交集的面积（如果两个框不相交则为 0）
            float w = std::max(0.0f, xx2 - xx1); // 交集宽度
            float h = std::max(0.0f, yy2 - yy1); // 交集高度
            float inter = w * h;                 // 交集面积

            // 计算两个框的并集面积，用于计算 IoU（Intersection over Union）
            float area1 = (det.x2 - det.x1) * (det.y2 - det.y1); // 框 1 的面积
            float area2 = (all_bboxes[idx2 * 4 + 2] - all_bboxes[idx2 * 4]) *
                          (all_bboxes[idx2 * 4 + 3] - all_bboxes[idx2 * 4 + 1]); // 框 2 的面积
            float ovr = inter / (area1 + area2 - inter);                         // IoU = 交集 / 并集

            // 如果 IoU 超过阈值，说明两个框重叠过多，应抑制低分框
            if (ovr >= nms_threshold)
            {
                suppressed[idx2] = true;
            }
        }
    }

    return results;
}
