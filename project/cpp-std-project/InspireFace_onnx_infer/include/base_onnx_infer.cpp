#include "base_onnx_infer.h"

BaseOnnxInfer::BaseOnnxInfer(const std::string &model_path, int intra_threads, bool use_gpu, int device_id)
{
    // 初始化成员变量
    m_model_path = model_path;
    m_intra_threads = intra_threads;
    m_use_gpu = use_gpu;
    m_deivce_id = device_id;

    // 初始化 session
    init_session();
    // 初始化模型
    init_model();
}


void BaseOnnxInfer::init_session()
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

void BaseOnnxInfer::init_model()
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
