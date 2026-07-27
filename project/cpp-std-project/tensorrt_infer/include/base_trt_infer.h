#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include "common.h"


namespace fs = std::filesystem;

class Logger final : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kINFO) {
			std::cout << "[TensorRT] " << msg << std::endl;
		}
	}
};
extern Logger gLogger;

template <typename T>
struct TrtDeleter {
	void operator()(T* ptr) const noexcept {
		if (ptr) {
			delete ptr;
		}
	}
};

struct CudaDeleter {
	void operator()(void* ptr) const noexcept {
		if (ptr) {
			cudaFree(ptr);
		}
	}
};

struct OutputBinding {
	std::string name;
	nvinfer1::Dims dims{};
	nvinfer1::DataType type{nvinfer1::DataType::kFLOAT};
	size_t count = 0;
	size_t bytes = 0;
	std::unique_ptr<void, CudaDeleter> device{nullptr};
	std::vector<float> hostFloat;
};

class BaseTRTInfer {
public:
    BaseTRTInfer() = default;

    // 析构函数，释放资源
    ~BaseTRTInfer();
    // 加载TensorRT引擎文件
    bool loadEngine(const std::string& enginePath);
    // 设置CUDA设备ID, 默认为0
    void setDevice(int deviceId) { m_deviceId = deviceId; }
    // 设置输入和输出的维度，默认为3
    void setInputDim(int inputDim) { m_input_dim = inputDim; }
    // 设置输出的维度，默认为3
    void setOutputDim(int outputDim) { m_output_dim = outputDim; }

    void _infer();

protected:

    int m_deviceId = 0;
    int m_input_dim = 3;
    int m_output_dim = 3;

	cudaStream_t m_stream = nullptr;
	std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> m_runtime;
	std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> m_engine;
	std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> m_context;
	std::unique_ptr<void, CudaDeleter> m_inputDevice{nullptr};
	std::string m_inputName;
	size_t m_inputBytes = 0;
	nvinfer1::Dims m_inputDims;
	nvinfer1::DataType m_inputType = nvinfer1::DataType::kFLOAT;
	nvinfer1::Dims m_outputDims;
	std::vector<OutputBinding> m_outputBindings;
	std::vector<std::uint8_t> m_hostInput;

    // 计算张量的体积（元素总数）
    size_t volume(const nvinfer1::Dims& dims) const;
    // 计算不同数据类型的元素大小
    size_t elementSize(nvinfer1::DataType type) const;
};