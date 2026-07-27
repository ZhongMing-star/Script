#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#include "curl_llm.h"

int main()
{

    LLMClient client("XXX", "XXX");
    BodyStruct body;

    body.model = "Qwen3.5-27B";
    body.messages = json::array({{{"role", "user"},
                                  {"content", "你是谁"}}});

    client.chat(body);
    return 0;
}
