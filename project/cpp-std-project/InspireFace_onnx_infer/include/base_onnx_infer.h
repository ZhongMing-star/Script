#pragma once

#include <iostream>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <vector>

class BaseOnnxInfer
{
public:
    BaseOnnxInfer(const std::string &model_path, int intra_threads, bool use_gpu, int device_id);
    ~BaseOnnxInfer() = default;

    std::string m_model_path;
    bool m_use_gpu;
    int m_deivce_id;
    int m_intra_threads;
    
    Ort::Env m_env{ORT_LOGGING_LEVEL_WARNING, "SCRFD_WITH_PRE"};
    Ort::Session m_session{nullptr};
    Ort::SessionOptions m_session_options;
    std::vector<Ort::AllocatedStringPtr> m_input_names_ptr;
    std::vector<Ort::AllocatedStringPtr> m_output_names_pt_;
    std::vector<const char*> m_input_names;
    std::vector<const char*> m_output_names;

    void init_session();
    void init_model();
};
