#ifndef CURL_LLM_H
#define CURL_LLM_H

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

// 请求体结构体，统一封装入参
struct BodyStruct
{
    std::string model;
    json messages;
    int max_tokens = 100;
    float temperature = 0.1f;
    float top_p = 0.95f;
    float presence_penalty = 0.0f;
    // 扩展字段会覆盖同名默认字段
    json extra_body = json::object();
    // 请求总超时，单位毫秒
    long timeout_ms = 30000L;
};

class LLMClient
{
public:
    LLMClient(std::string base_url, std::string api_key);

    // 对外主接口
    json chat(const BodyStruct& body);

private:
    // 拼接请求json字符串
    std::string build_body(const BodyStruct& body);

    std::string m_base_url;
    std::string m_api_key;
};

#endif