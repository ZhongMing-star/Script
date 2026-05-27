#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class MinimalSubscriber : public rclcpp::Node
{
public:
    MinimalSubscriber ()
        : Node("minimal_subscriber")
    {
        // 创建订阅者
        subscriber_ = this->create_subscription<std_msgs::msg::String>("minimal_topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "C++ 订阅者已启动，正在接收消息...");
    }

private:
    void topic_callback(const std_msgs::msg::String::SharedPtr msg);
       rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscriber_;
};


void MinimalSubscriber::topic_callback(const std_msgs::msg::String::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), "Received: '%s'", msg->data.c_str());
}


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MinimalSubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
