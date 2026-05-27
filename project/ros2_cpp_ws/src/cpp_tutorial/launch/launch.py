from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 启动 C++ 发布者
        Node(
            package="cpp_tutorial",
            executable="publisher",
            name="minimal_publisher",
            output="screen"
        ),
        
        # 启动 C++ 订阅者
        Node(
            package="cpp_tutorial",
            executable="subscriber",
            name="minimal_subscriber",
            output="screen"
        ),
    ])