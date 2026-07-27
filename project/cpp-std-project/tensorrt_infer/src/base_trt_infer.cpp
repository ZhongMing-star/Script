#include "base_trt_infer.h"

Logger gLogger;

BaseTRTInfer::~BaseTRTInfer(){
    if (m_stream != nullptr) {
        cudaStreamDestroy(m_stream);
    }    
}

// 计算张量的体积（元素总数）
size_t BaseTRTInfer::volume(const nvinfer1::Dims& dims) const {
    size_t vol = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        vol *= dims.d[i];
    }
    return vol;
}

// 计算不同数据类型的元素大小
size_t BaseTRTInfer::elementSize(nvinfer1::DataType type) const {
    switch (type) {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
        return sizeof(std::uint16_t);
    case nvinfer1::DataType::kINT8:
        return sizeof(std::int8_t);
    case nvinfer1::DataType::kINT32:
        return sizeof(std::int32_t);
    case nvinfer1::DataType::kUINT8:
        return sizeof(std::uint8_t);
    case nvinfer1::DataType::kBOOL:
        return sizeof(bool);
    default:
        throw std::runtime_error("未知 TensorRT DataType");
    }
}

bool BaseTRTInfer::loadEngine(const std::string& enginePath){

    std::ifstream file(enginePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开引擎文件, 引擎文件路径: " + enginePath);
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(static_cast<size_t>(size));
    if (!file.read(engineData.data(), size)) {
        throw std::runtime_error("读取引擎文件失败, 引擎文件路径: " + enginePath);
    }

    if (cudaSetDevice(m_deviceId) != cudaSuccess) {
        throw std::runtime_error("设置 CUDA 设备失败");
    }

    initLibNvInferPlugins(&gLogger, "");
    m_runtime.reset(nvinfer1::createInferRuntime(gLogger));
    if (!m_runtime) {
        throw std::runtime_error("创建 TensorRT runtime 失败");
    }

    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!m_engine) {
        throw std::runtime_error("反序列化 TensorRT 引擎失败");
    }

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        throw std::runtime_error("创建执行上下文失败");
    }

    const int tensorCount = m_engine->getNbIOTensors();
    m_outputBindings.clear();
    for (int i = 0; i < tensorCount; ++i) {
        const char* name = m_engine->getIOTensorName(i);
        // 隐含前提：模型只有 1 个输入张量，多输入模型会被覆盖，只保留最后遍历到的输入
        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            m_inputName = name;
        } else {
            OutputBinding binding;
            binding.name = name;
            m_outputBindings.push_back(std::move(binding));
        }
    }

    if (m_inputName.empty() || m_outputBindings.empty()) {
        throw std::runtime_error("未找到输入或输出 tensor");
    }

    m_inputDims = m_engine->getTensorShape(m_inputName.c_str());
    m_outputDims = m_engine->getTensorShape(m_outputBindings[0].name.c_str());
    m_inputType = m_engine->getTensorDataType(m_inputName.c_str());

    if(m_inputDims.nbDims != m_input_dim || m_outputDims.nbDims != m_output_dim) {
        throw std::runtime_error("输入或输出 tensor 的维度不为预期值, 输入维度: " + std::to_string(m_inputDims.nbDims) + ", 输出维度: " + std::to_string(m_outputDims.nbDims));
    }
    
    if (!m_context->setInputShape(m_inputName.c_str(), m_inputDims)) {
        throw std::runtime_error("设置输入 tensor 形状失败");
    }

    for (auto &binding : m_outputBindings) {
        binding.dims = m_engine->getTensorShape(binding.name.c_str());
        if (binding.dims.nbDims != m_output_dim) {
            throw std::runtime_error("输出 tensor 的维度不为预期值, name=" + binding.name);
        }
        binding.type = m_engine->getTensorDataType(binding.name.c_str());
        binding.count = volume(binding.dims);
        binding.bytes = binding.count * sizeof(float);
        binding.hostFloat.assign(binding.count, 0.0f);
    }

    m_outputDims = m_context->getTensorShape(m_outputBindings[0].name.c_str());
    if (m_outputDims.nbDims != m_output_dim) {
        throw std::runtime_error("获取输出 tensor 维度失败");
    }

    size_t inputCount = volume(m_inputDims);
    m_inputBytes = inputCount * elementSize(m_inputType);
    m_hostInput.assign(m_inputBytes, 0);

    void* inputDevicePtr = nullptr;
    // void* outputDevicePtr = nullptr;
    if(cudaStreamCreate(&m_stream) != cudaSuccess) {
        throw std::runtime_error("创建 CUDA 流失败");
    }
    // 默认只有一个输入
    if (cudaMalloc(&inputDevicePtr, m_inputBytes) != cudaSuccess) {
        throw std::runtime_error("分配输入 tensor 的 GPU 内存失败");
    }
    m_inputDevice.reset(inputDevicePtr);

    for(size_t i = 0; i < m_outputBindings.size(); ++i) {
        OutputBinding& binding = m_outputBindings[i];
        binding.dims = m_engine->getTensorShape(binding.name.c_str());
        if (binding.dims.nbDims != m_output_dim) {
            throw std::runtime_error("输出 tensor 的维度不为预期值, name=" + binding.name);
        }
        binding.type = m_engine->getTensorDataType(binding.name.c_str());
        binding.count = volume(binding.dims);
        binding.bytes = binding.count * sizeof(float);
        binding.hostFloat.assign(binding.count, 0.0f);

        void* outputDevicePtr = nullptr;
        if (cudaMalloc(&outputDevicePtr, binding.bytes) != cudaSuccess) {
            throw std::runtime_error("分配输出 tensor 的 GPU 内存失败, name=" + binding.name);
        }
        binding.device.reset(outputDevicePtr);
    }

    return true;
}

void BaseTRTInfer::_infer() {
    if (!m_context) {
        throw std::runtime_error("执行上下文未初始化");
    }
    if (cudaMemcpyAsync(m_inputDevice.get(), m_hostInput.data(), m_inputBytes, cudaMemcpyHostToDevice, m_stream) != cudaSuccess) {
        throw std::runtime_error("将输入数据从主机复制到设备失败");
    }
    
    if (!m_context->setTensorAddress(m_inputName.c_str(), m_inputDevice.get())) {
        throw std::runtime_error("设置输入 tensor 地址失败");
    }

    for (const auto& binding : m_outputBindings) {
        if (!m_context->setTensorAddress(binding.name.c_str(), binding.device.get())) {
            throw std::runtime_error("设置输出 tensor 地址失败: " + binding.name);
        }
    }

    if (!m_context->enqueueV3(m_stream)) {
        throw std::runtime_error("TensorRT 推理执行失败");
    }

    for(auto &binding : m_outputBindings) {
        if (binding.type != nvinfer1::DataType::kFLOAT || binding.hostFloat.empty()) {
            continue;
        }
        if (cudaMemcpyAsync(binding.hostFloat.data(), binding.device.get(), binding.bytes, cudaMemcpyDeviceToHost, m_stream) != cudaSuccess) {
            throw std::runtime_error("将输出数据从设备复制到主机失败: " + binding.name);
        }
    }

    if (cudaStreamSynchronize(m_stream) != cudaSuccess) {
        throw std::runtime_error("同步 CUDA 流失败");
    }
}