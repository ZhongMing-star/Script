#include "inspire_face_demo_node.cpp"


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InspireFaceDemoNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}