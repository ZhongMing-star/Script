#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class MinimalPublisher  : public rclcpp::Node
{
public:
    MinimalPublisher()
        : Node("minimal_publisher")
    {
        // 创建发布者
        publisher_ = this->create_publisher<std_msgs::msg::String>("minimal_topic", 10);

        // 创建定时器
        timer_ = this->create_wall_timer( 500ms, std::bind(&MinimalPublisher::timer_callback, this));


        RCLCPP_INFO(this->get_logger(), "C++ 发布者已启动，正在发送消息...");
    }

private:
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    void timer_callback();
};


void MinimalPublisher::timer_callback()
{
    auto message = std_msgs::msg::String();
    message.data = "Hello, world!";

    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
}


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MinimalPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
