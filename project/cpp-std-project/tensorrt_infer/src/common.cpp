#include "common.h"

#include <cctype>
#include <numeric>

namespace fs = std::filesystem;


// Sigmoid 函数
float VisionCore::sigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

// 计算两个矩形框的IoU（Intersection over Union）
float VisionCore::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
	const float x1 = std::max(a.x, b.x);
	const float y1 = std::max(a.y, b.y);
	const float x2 = std::min(a.x + a.width, b.x + b.width);
	const float y2 = std::min(a.y + a.height, b.y + b.height);
	const float interW = std::max(0.0f, x2 - x1);
	const float interH = std::max(0.0f, y2 - y1);
	const float interArea = interW * interH;
	const float areaA = std::max(0.0f, a.width) * std::max(0.0f, a.height);
	const float areaB = std::max(0.0f, b.width) * std::max(0.0f, b.height);
	return interArea / (areaA + areaB - interArea + 1e-6f);
}

// 非极大值抑制（Non-Maximum Suppression, NMS）算法
std::vector<VisionCore::Detection2D> VisionCore::nms(const std::vector<VisionCore::Detection2D>& detections, float iouThreshold) {
		std::vector<VisionCore::Detection2D> result;
		if (detections.empty()) {
			return result;
		}

		std::vector<int> order(detections.size());
		std::iota(order.begin(), order.end(), 0);
		std::sort(order.begin(), order.end(), [&detections](int lhs, int rhs) {
			return detections[static_cast<size_t>(lhs)].conf > detections[static_cast<size_t>(rhs)].conf;
		});

		std::vector<bool> removed(detections.size(), false);
		for (size_t i = 0; i < order.size(); ++i) {
			const int current = order[i];
			if (removed[static_cast<size_t>(current)]) {
				continue;
			}
			result.push_back(detections[static_cast<size_t>(current)]);
			for (size_t j = i + 1; j < order.size(); ++j) {
				const int other = order[j];
				if (removed[static_cast<size_t>(other)]) {
					continue;
				}
				if (detections[static_cast<size_t>(current)].cls_id != detections[static_cast<size_t>(other)].cls_id) {
					continue;
				}
				if (iou(detections[static_cast<size_t>(current)].bbox, detections[static_cast<size_t>(other)].bbox) > iouThreshold) {
					removed[static_cast<size_t>(other)] = true;
				}
			}
		}

		return result;
	}

// 对图像进行Letterbox缩放和填充，使其适应目标尺寸，同时保持纵横比
VisionCore::LetterBoxInfo VisionCore::letterbox(const cv::Mat& image, cv::Mat& output, int targetSize) {
	const float scale = std::min(static_cast<float>(targetSize) / static_cast<float>(image.cols),
								 static_cast<float>(targetSize) / static_cast<float>(image.rows));
	const int resizedWidth = static_cast<int>(std::round(image.cols * scale));
	const int resizedHeight = static_cast<int>(std::round(image.rows * scale));
	const int padW = targetSize - resizedWidth;
	const int padH = targetSize - resizedHeight;
	const int padLeft = 0;
	const int padTop = 0;
	const int padRight = padW - padLeft;
	const int padBottom = padH - padTop;

	cv::Mat resized;
	cv::resize(image, resized, cv::Size(resizedWidth, resizedHeight), 0, 0, cv::INTER_LINEAR);
	cv::copyMakeBorder(resized, output, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	return {scale, padLeft, padTop, image.cols, image.rows};
}

/// 判断文件是否为图像文件
bool VisionCore::isImageFile(const fs::path& path) {
	if (!path.has_extension()) {
		return false;
	}
	std::string ext = path.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
		return static_cast<char>(std::tolower(ch));
	});
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff" || ext == ".webp";
}

void VisionCore::rotate_image(const cv::Mat &image, int angle)
{
    switch (angle)
    {
    case 90:
        cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
        break;
    case 180:
        cv::rotate(image, image, cv::ROTATE_180);
        break;
    case 270:
        cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
        break;
    default:
        // 任意角度旋转
        cv::Point2f center(image.cols / 2.0, image.rows / 2.0);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, -angle, 1.0);
        cv::warpAffine(image, image, rot_mat, image.size());
        break;
    }
}