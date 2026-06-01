#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;


int main() {
    // 1. 创建一个 JSON 对象 
    json j;
    j["name"] = "Alice";
    j["age"] = 30;
    j["sources"] = {"C++", "Python", "JavaScript"};
    j["user"] = {
        {"id", 1},
        {"username", "alice123"}
    };

    // 2. 输出 JSON 对象
    cout << "JSON Object:" << endl;
    cout << j.dump(4) << endl; // 以漂亮的格式输出 JSON 对象

    // 3. 访问 JSON 对象的属性
    cout << "Name: " << j["name"] << endl;
    cout << "Age: " << j["age"] << endl;
    cout << "Sources: " << j["sources"][0] << endl;
    cout << "User ID: " << j["user"]["id"] << endl;
    cout << "User Username: " << j["user"]["username"] << endl;

    // 4. 修改 JSON 对象的属性
    j["age"] = 31;
    cout << "Updated Age: " << j["age"] << endl;

    // 5. 删除 JSON 对象的属性
    j.erase("sources");
    cout << "After deleting sources:" << endl;
    cout << j.dump(4) << endl;

    // 6. 添加 JSON 对象的属性
    j["email"] = "alice@example.com";
    cout << "After adding email:" << endl;
    cout << j.dump(4) << endl;

    // 7. 解析 JSON 字符串
    json j2;
    try{
        j2 = json::parse(j.dump());
    }
    catch(const json::exception& e){
        cerr << "JSON Parse Error: " << e.what() << endl;
        cerr << "Failed JSON string: " << j.dump() << endl;
        cerr << "Error Message: " << e.what() << endl;
    }
    catch(const std::exception& e){
        cerr << "Exception: " << e.what() << endl;
    }
    cout << "JSON Parse Success" << endl;
    cout << j2.dump(4) << endl;

    // 8. 常用 api 
    if (j.empty()) {
        cout << "JSON object is empty." << endl;
    } else {
        cout << "JSON object is not empty." << endl;
    }

    cout << "JSON Object Size: " << j.size() << endl;
    cout << "JSON erase Keys: " << j.erase("name") << endl;
    cout << "JSON contains Keys: " << j.contains("name") << endl;

    cout << "JSON age is_null() : " << j["age"].is_null() << endl;
    cout << "JSON age is_array() : " << j["age"].is_array() << endl;
    cout << "JSON age is_string() : " << j["age"].is_string() << endl;
    
    cout << "JSON age is_number() : " << j["age"].is_number() << endl;
    cout << "JSON age: " << j["age"].get<double>() << endl;

    j["sources"].push_back("Go");
    j["sources"].push_back("Python");
    j["sources"].push_back("Cpp");
    cout << "JSON sources: " << j["sources"] << endl;

    
    return 0;
}