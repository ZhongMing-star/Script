#include "curl_llm.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;
namespace
{
    void ensure_curl_global_init()
    {
        static std::once_flag once;
        static CURLcode init_result = CURLE_OK;

        std::call_once(once, []() {
            init_result = curl_global_init(CURL_GLOBAL_ALL);
        });

        if (init_result != CURLE_OK)
        {
            throw std::runtime_error(
                "curl global init failed: " + std::string(curl_easy_strerror(init_result)));
        }
    }

    struct CurlEasyDeleter
    {
        void operator()(CURL* handle) const noexcept
        {
            if (handle != nullptr)
            {
                curl_easy_cleanup(handle);
            }
        }
    };

    struct CurlSlistDeleter
    {
        void operator()(curl_slist* list) const noexcept
        {
            if (list != nullptr)
            {
                curl_slist_free_all(list);
            }
        }
    };

    using CurlEasyPtr = std::unique_ptr<CURL, CurlEasyDeleter>;
    using CurlSlistPtr = std::unique_ptr<curl_slist, CurlSlistDeleter>;
}

LLMClient::LLMClient(std::string base_url, std::string api_key)
    : m_base_url(std::move(base_url)), m_api_key(std::move(api_key))
{
}

// curl接收数据回调
static size_t write_callback(char *ptr, size_t size, size_t nmemb, std::string *out_buf)
{
    if (out_buf == nullptr)
    {
        return 0;
    }
    out_buf->append(ptr, size * nmemb);
    return size * nmemb;
}

// 组装POST请求JSON字符串
std::string LLMClient::build_body(const BodyStruct &body)
{
    json req_json;
    req_json["model"] = body.model;
    req_json["messages"] = body.messages;
    req_json["max_tokens"] = body.max_tokens;
    req_json["temperature"] = body.temperature;
    req_json["top_p"] = body.top_p;
    req_json["presence_penalty"] = body.presence_penalty;

    if (!body.extra_body.is_object())
    {
        throw std::runtime_error("extra_body must be a JSON object");
    }

    // 合并扩展字段
    req_json.update(body.extra_body);

    return req_json.dump();
}

json LLMClient::chat(const BodyStruct& body)
{
    ensure_curl_global_init();

    CurlEasyPtr curl(curl_easy_init());
    if (!curl)
    {
        throw std::runtime_error("curl_easy_init failed, create curl handle error");
    }

    if (m_base_url.empty())
    {
        throw std::runtime_error("base_url is empty");
    }

    if (body.model.empty())
    {
        throw std::runtime_error("model is empty");
    }

    std::string response_buf;
    std::string post_str = build_body(body);
    std::array<char, CURL_ERROR_SIZE> errbuf{};

    // 构造请求头
    curl_slist *raw_headers = nullptr;
    raw_headers = curl_slist_append(raw_headers, "Content-Type: application/json");
    raw_headers = curl_slist_append(raw_headers, "Accept: application/json");
    raw_headers = curl_slist_append(raw_headers, "Expect:");
    std::string auth_header = "Authorization: Bearer " + m_api_key;
    raw_headers = curl_slist_append(raw_headers, auth_header.c_str());
    CurlSlistPtr headers(raw_headers);
    if (!headers)
    {
        throw std::runtime_error("failed to create curl headers");
    }

    const long timeout_ms = std::max(1L, body.timeout_ms);
    const long connect_timeout_ms = std::min(10000L, timeout_ms);

    // curl 全局配置
    curl_easy_setopt(curl.get(), CURLOPT_URL, m_base_url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_POST, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, post_str.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE, static_cast<long>(post_str.size()));
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());
    curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_ERRORBUFFER, errbuf.data());

    // 接收返回数据回调
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &response_buf);

    // 设置超时时间
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS, connect_timeout_ms);
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS, timeout_ms);

    // 发起网络请求
    CURLcode curl_ret = curl_easy_perform(curl.get());

    // 错误分类判断
    if (curl_ret == CURLE_OPERATION_TIMEDOUT)
    {
        throw std::runtime_error("curl request failed: timeout reached");
    }
    if (curl_ret != CURLE_OK)
    {
        std::ostringstream oss;
        oss << "curl request failed: " << curl_easy_strerror(curl_ret);
        if (errbuf[0] != '\0')
        {
            oss << " | detail: " << errbuf.data();
        }
        throw std::runtime_error(oss.str());
    }

    long http_code = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code < 200 || http_code >= 300)
    {
        std::ostringstream oss;
        oss << "http request failed, status=" << http_code;
        if (!response_buf.empty())
        {
            oss << " | response=" << response_buf;
        }
        throw std::runtime_error(oss.str());
    }

    // 空响应拦截，避免json解析崩溃
    if (response_buf.empty())
    {
        throw std::runtime_error("LLM server return empty response string");
    }
    
    try
    {
        return json::parse(response_buf);
    }
    catch (const json::parse_error &e)
    {
        std::string err_info = "json parse failed, raw response: " + response_buf + " | err: " + std::string(e.what());
        throw std::runtime_error(err_info);
    }
}