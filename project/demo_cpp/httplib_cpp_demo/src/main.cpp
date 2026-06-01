#include <iostream>
#include "cpp-httplib/httplib.h"
#include <yaml-cpp/yaml.h>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

int main() {
    std::cout << "Hello, World!" << std::endl;

    httplib::Server svr;

    svr.Get("/hi", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("Hello World!", "text/plain");
    });

    svr.Get("/json", [](const httplib::Request& req, httplib::Response& res) {
        json j;
        j["code"] = 0;
        j["msg"] = "success";
        res.set_content(j.dump(), "application/json");
    });

    std::cout << "Server running on http://0.0.0.0:8080" << std::endl;
    svr.listen("0.0.0.0", 8080);

    return 0;
}