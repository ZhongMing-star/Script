#include <iostream>
#include <yaml-cpp/yaml.h>
using namespace std;

int main() {
    // 1. 加载
    YAML::Node cfg = YAML::Load(R"(
            server: { port: 8080, ips: [192.168.1.1, 192.168.1.2]
            }
        )");

    // 2. 取值
    int port = cfg["server"]["port"].as<int>();
    cout << "port: " << port << endl;

    // 3. 遍历数组
    for (const auto& ip : cfg["server"]["ips"]) {
        cout << "ip: " << ip.as<string>() << endl;
    }

    // 4. 新增
    cfg["server"]["name"] = "test";

    // 5. 输出
    cout << "\n" << YAML::Dump(cfg) << endl;
}