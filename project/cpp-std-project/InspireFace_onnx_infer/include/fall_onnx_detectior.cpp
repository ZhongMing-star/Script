#include "fall_onnx_detectior.h"

FallOnnxDetector::FallOnnxDetector(const std::string &model_path, float threshold, float nms_threshold, int intra_threads, bool use_gpu, int device_id)
    : BaseOnnxInfer(model_path, intra_threads, use_gpu, device_id)
{
    // 初始化成员变量
    m_threshold = threshold;
    m_nms_threshold = nms_threshold;
}

std::vector<DetectInfo> FallOnnxDetector::detect(const cv::Mat& image)
{

    if (image.empty())
    {
        std::cerr << "[ERROR] Empty input image" << std::endl;
        return {};
    }

    // 第一步： 前处理
    float scale;
    this->preprocess(image, scale);

    // 第二步： 构造推理张量
    std::vector<int64_t> input_dims = {image.rows, image.cols, 3};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto input_mem = Ort::Value::CreateTensor<uint8_t>(
        memory_info,
        image.data,
        image.total() * image.channels(),
        input_dims.data(),   // 张量维度数组
        input_dims.size()    // 维度数
    );
    // 第三步：执行模型推理
    std::vector<Ort::Value> outputs = m_session.Run(
        Ort::RunOptions{},     // 运行选项（使用默认设置）
        m_input_names.data(),  // 输入名称数组
        &input_mem,            // 输入张量数组
        m_input_names.size(),  // 输入数量
        m_output_names.data(), // 输出名称数组
        m_output_names.size()  // 输出数量
    );

    // 第四步：后处理模型输出，生成最终的检测结果
    std::vector<DetectInfo> res = postprocess(outputs, scale, image.size()); // 后处理

    return res;
}

void FallOnnxDetector::preprocess(const cv::Mat& img, float &scale)
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
}

std::vector<DetectInfo> FallOnnxDetector::postprocess(std::vector<Ort::Value>& output, float& scale, cv::Size input_size)
{
    std::vector<DetectInfo> result;
    // YOLOv8 ONNX output0 shape: [1, num_attr, 8400]
    Ort::Value& out_tensor = output[0];
    auto shape_info = out_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> out_dims = shape_info.GetShape();

    const int batch = static_cast<int>(out_dims[0]);
    const int num_attr = static_cast<int>(out_dims[1]);
    const int num_box = static_cast<int>(out_dims[2]);
    const int num_cls = num_attr - 4;

    // 获取浮点数据指针
    float* data_ptr = (float*)out_tensor.GetTensorMutableData<float>();
    // 取batch=0的数据，跳过batch维度
    float* pred = data_ptr;

    // input_size: w_model, h_model
    int w_model = input_size.width;
    int h_model = input_size.height;
    // scale = 原图宽 / 模型输入宽 = w_orig / w_model
    float w_orig = w_model / scale;
    float h_orig = h_model / scale;

    // 遍历全部8400个候选框
    for (int box_idx = 0; box_idx < num_box; box_idx++)
    {
        // 取出xywh: pred[0*num_box + box_idx] ... pred[3*num_box + box_idx]
        float cx = pred[0 * num_box + box_idx];
        float cy = pred[1 * num_box + box_idx];
        float w = pred[2 * num_box + box_idx];
        float h = pred[3 * num_box + box_idx];

        // 遍历所有类别，找最大置信度
        float max_score = 0.0f;
        int max_cls_id = 0;
        for (int c = 0; c < num_cls; c++)
        {
            float score = pred[(4 + c) * num_box + box_idx];
            if (score > max_score)
            {
                max_score = score;
                max_cls_id = c;
            }
        }
        // 过滤低置信度框
        if (max_score < m_threshold)
            continue;

        // YOLO cxcywh -> 模型图xyxy
        float x1_model = cx - w / 2.0f;
        float y1_model = cy - h / 2.0f;
        float x2_model = cx + w / 2.0f;
        float y2_model = cy + h / 2.0f;

        // 映射回原图坐标（scale = orig_w / model_w）
        int x1 = std::max(0, (int)(x1_model / scale));
        int y1 = std::max(0, (int)(y1_model / scale));
        int x2 = std::min((int)w_orig, (int)(x2_model / scale));
        int y2 = std::min((int)h_orig, (int)(y2_model / scale));

        // 构造检测框
        DetectInfo info;
        info.bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        info.score = max_score;
        info.id = max_cls_id;
        result.push_back(info);
    }

    // 执行NMS过滤
    nms(result);
    return result;
}

void FallOnnxDetector::nms(std::vector<DetectInfo>& detect_info)
{
    if (detect_info.empty())
        return;

    // 1. 按类别分组：key=class_id, value=当前类所有检测框下标
    std::unordered_map<int, std::vector<int>> cls_group;
    for (int idx = 0; idx < (int)detect_info.size(); idx++)
    {
        int cls = detect_info[idx].id;
        cls_group[cls].push_back(idx);
    }

    std::vector<int> keep_indices; // 最终保留的检测框下标

    // 遍历每个类别单独做NMS
    for (auto& group_pair : cls_group)
    {
        auto& cls_indices = group_pair.second;
        // 按置信度从高到低排序
        std::sort(cls_indices.begin(), cls_indices.end(), [&](int a, int b) {
            return detect_info[a].score > detect_info[b].score;
        });

        std::vector<int> cls_keep;
        while (!cls_indices.empty())
        {
            // 取当前最高分框
            int top_idx = cls_indices[0];
            cls_keep.push_back(top_idx);
            cv::Rect box_a = detect_info[top_idx].bbox;

            // 剩余框计算IOU，过滤大于阈值的框
            std::vector<int> new_indices;
            for (size_t j = 1; j < cls_indices.size(); j++)
            {
                int curr_idx = cls_indices[j];
                cv::Rect box_b = detect_info[curr_idx].bbox;

                // 计算相交区域
                int inter_x1 = std::max(box_a.x, box_b.x);
                int inter_y1 = std::max(box_a.y, box_b.y);
                int inter_x2 = std::min(box_a.x + box_a.width, box_b.x + box_b.width);
                int inter_y2 = std::min(box_a.y + box_a.height, box_b.y + box_b.height);

                int inter_w = std::max(0, inter_x2 - inter_x1);
                int inter_h = std::max(0, inter_y2 - inter_y1);
                float inter_area = (float)inter_w * inter_h;

                float area_a = (float)box_a.width * box_a.height;
                float area_b = (float)box_b.width * box_b.height;
                float iou = inter_area / (area_a + area_b - inter_area);

                if (iou <= m_nms_threshold)
                    new_indices.push_back(curr_idx);
            }
            cls_indices.swap(new_indices);
        }
        keep_indices.insert(keep_indices.end(), cls_keep.begin(), cls_keep.end());
    }

    // 根据保留下标重构结果
    std::vector<DetectInfo> res;
    for (int idx : keep_indices)
        res.push_back(detect_info[idx]);
    detect_info.swap(res);
}