#pragma once

#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>

namespace VisionCore{
	struct Detection2D  {
		cv::Rect2f bbox;
		int cls_id = -1;
		float conf  = 0.0f;
	};
	
	struct LetterBoxInfo {
		float scale = 1.0f;
		int padLeft = 0;
		int padTop = 0;
		int originalWidth = 0;
		int originalHeight = 0;
	};

	
	struct FaceDetection2D  {
		cv::Rect2f bbox;
		int cls_id = -1;
		float conf  = 0.0f;
		std::vector<cv::Point2f> landmarks;
	};
	
	
	
	// Sigmoid 函数
	float sigmoid(float x);
	// 计算两个矩形框的IoU（Intersection over Union）
	float iou(const cv::Rect2f& a, const cv::Rect2f& b);
	// 对图像进行Letterbox缩放和填充，使其适应目标尺寸，同时保持纵横比
	LetterBoxInfo letterbox(const cv::Mat& image, cv::Mat& output, int targetSize);
	// 判断文件是否为图像文件
	bool isImageFile(const std::filesystem::path& path);
	// 非极大值抑制（NMS）算法，用于去除重叠的检测框
	std::vector<Detection2D> nms(const std::vector<Detection2D>& detections, float iouThreshold);
	// 将图像旋转指定角度
	void rotate_image(const cv::Mat &image, int angle);
}
