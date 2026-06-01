#pragma once

#include <iostream>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "face_common.hpp"
#include "inspireface_wrapper.hpp"

class InspireFaceDemoNode: public rclcpp::Node
{
public:
    InspireFaceDemoNode();
    ~InspireFaceDemoNode();

private:
    std::shared_ptr<InspireFaceFeatureExtract> m_face_feature_extract;
    std::shared_ptr<InspireFaceDetector> m_face_detect;

};

InspireFaceDemoNode::InspireFaceDemoNode() : Node("inspire_face_demo_node") {
    // 初始化人脸识别器
    std::string resource_path = ament_index_cpp::get_package_share_directory("inspire_face_demo_cpp") + "/resources";
    std::string model_pack_path  = resource_path + "/models/Pikachu";
    RCLCPP_INFO(this->get_logger(), "model_pack_path: %s", model_pack_path.c_str());

    InspireFaceGlobal::getInstance().Init(model_pack_path);
    m_face_feature_extract = std::make_shared<InspireFaceFeatureExtract>();
    m_face_detect = std::make_shared<InspireFaceDetector>();
}

InspireFaceDemoNode::~InspireFaceDemoNode()
{
}