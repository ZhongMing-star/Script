#include "openai.hpp"

#include <iostream>

int main() {

    // 自定义Session配置：关闭SSL证书校验
    openai::start(
        "dyg-dcDSMsuJJG5KUYWopbacplj37UFBq2K33KPLCiNQsfk",
        "",
        false,
        "http://xai.dyg.com.cn/v1/"
    );

    // 多模态请求，图片填网络地址
    auto chat = openai::chat().create(R"(
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
        "extra_body":{
            "chat_template_kwargs": {
                "enable_thinking": false
            }
        }
    }
    )"_json);
    std::cout << "Response2 is:\n" << chat.dump(2) << '\n'; 
}