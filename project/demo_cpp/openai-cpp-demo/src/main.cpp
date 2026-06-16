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
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述图中内容"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://gips3.baidu.com/it/u=1039279337,1441343044&fm=3028&app=3028&f=JPEG&fmt=auto&q=100&size=f1024_1024"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0
    }
    )"_json);
    std::cout << "Response2 is:\n" << chat.dump(2) << '\n'; 
}