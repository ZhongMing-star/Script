#define NLOHMANN_JSON_USE_USER_LITERALS
#include "openai.hpp"
#include <iostream>
#include <curl/curl.h>
#include <string>

static size_t writeCb(char* ptr, size_t s, size_t n, std::string* buf) {
    buf->append(ptr, s * n);
    return s * n;
}

int main() {
    CURL* curl = curl_easy_init();
    std::string resp;
    std::string req_body = R"(
    {
        "model": "Qwen3.5-27B",
        "messages": [
            {
                "role": "system",
                "content": "你是湖北光谷东智具身智能技术有限公司研发的机器人助手，你的名字叫 **光子**。你的角色设定是：友好、有耐心、轻度幽默、善于倾听，能够进行自然流畅的日常聊天。"
            },
            {
                "role": "user",
                "content": "你是谁?"
            }
        ],
        "max_tokens": 1000,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": false}
    }
    )";

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Authorization: Bearer dyg-dcDSMsuJJG5KUYWopbacplj37UFBq2K33KPLCiNQsfk");

    curl_easy_setopt(curl, CURLOPT_URL, "http://xai.dyg.com.cn/v1/chat/completions");
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, req_body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_perform(curl);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    nlohmann::json chat = nlohmann::json::parse(resp);
    std::cout << "Response2 is:\n" << chat.dump(2) << '\n';
    return 0;
}