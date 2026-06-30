#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace {

class Logger final : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kINFO) {
			std::cout << "[TensorRT] " << msg << std::endl;
		}
	}
};

Logger gLogger;

template <typename T>
struct TrtDeleter {
	void operator()(T* ptr) const noexcept {
		if (ptr) {
			delete ptr;
		}
	}
};

struct CudaDeleter {
	void operator()(void* ptr) const noexcept {
		if (ptr) {
			cudaFree(ptr);
		}
	}
};

struct Detection {
	cv::Rect2f box;
	int classId = -1;
	float score = 0.0f;
};

struct LetterBoxInfo {
	float scale = 1.0f;
	int padLeft = 0;
	int padTop = 0;
};

struct Arguments {
	std::string enginePath = "resource/model/fall_detection.trt";
	std::string imagePath;
	std::string imgRootPath = "img_root";
	std::string outputPath = "outputs/result.jpg";
	std::string labelsPath = "config/labels.txt";
	float confThreshold = 0.25f;
	float iouThreshold = 0.45f;
};

static inline float sigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
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

static std::vector<std::string> loadLabels(const std::string& labelPath, int classCount) {
	std::vector<std::string> labels;
	if (!labelPath.empty() && fs::exists(labelPath)) {
		std::ifstream fin(labelPath);
		std::string line;
		while (std::getline(fin, line)) {
			if (!line.empty()) {
				labels.push_back(line);
			}
		}
	}
	if (labels.empty()) {
		for (int i = 0; i < classCount; ++i) {
			labels.push_back("class_" + std::to_string(i));
		}
	}
	return labels;
}

static LetterBoxInfo letterbox(const cv::Mat& image, cv::Mat& output, int targetSize) {
	std::cout << "原始图片尺寸: " << image.cols << "x" << image.rows << std::endl;
	std::cout << "目标尺寸: " << targetSize << "x" << targetSize << std::endl;

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
	return {scale, padLeft, padTop};
}

static void printUsage(const char* program) {
	std::cout << "用法: " << program
			  << " [--img_root <image_root>] [--image <image_path>] [--engine <engine_path>] [--output <output_path>]"
			  << " [--labels <labels_path>] [--conf 0.25] [--iou 0.45]" << std::endl;
}

static Arguments parseArguments(int argc, char** argv) {
	Arguments args;
	for (int i = 1; i < argc; ++i) {
		const std::string key = argv[i];
		auto requireValue = [&](const std::string& option) -> std::string {
			if (i + 1 >= argc) {
				throw std::runtime_error("参数缺少值: " + option);
			}
			return argv[++i];
		};

		if (key == "--engine") {
			args.enginePath = requireValue(key);
		} else if (key == "--image") {
			args.imagePath = requireValue(key);
		} else if (key == "--img_root") {
			args.imgRootPath = requireValue(key);
		} else if (key == "--output") {
			args.outputPath = requireValue(key);
		} else if (key == "--labels") {
			args.labelsPath = requireValue(key);
		} else if (key == "--conf") {
			args.confThreshold = std::stof(requireValue(key));
		} else if (key == "--iou") {
			args.iouThreshold = std::stof(requireValue(key));
		} else if (key == "--help" || key == "-h") {
			printUsage(argv[0]);
			std::exit(0);
		} else {
			throw std::runtime_error("未知参数: " + key);
		}
	}
	return args;
}

static bool isImageFile(const fs::path& path) {
	if (!path.has_extension()) {
		return false;
	}
	std::string ext = path.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
		return static_cast<char>(std::tolower(ch));
	});
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff" || ext == ".webp";
}

static std::vector<fs::path> collectImages(const fs::path& root) {
	std::vector<fs::path> images;
	if (!fs::exists(root)) {
		return images;
	}
	if (fs::is_regular_file(root)) {
		if (isImageFile(root)) {
			images.push_back(root);
		}
		return images;
	}
	for (const auto& entry : fs::recursive_directory_iterator(root)) {
		if (entry.is_regular_file() && isImageFile(entry.path())) {
			images.push_back(entry.path());
		}
	}
	std::sort(images.begin(), images.end());
	return images;
}

class YoloTrtDetector {
public:
	~YoloTrtDetector() {
		if (stream_ != nullptr) {
			cudaStreamDestroy(stream_);
		}
	}

	bool loadEngine(const std::string& enginePath) {
		std::ifstream file(enginePath, std::ios::binary);
		if (!file) {
			std::cerr << "无法打开引擎文件: " << enginePath << std::endl;
			return false;
		}

		file.seekg(0, std::ios::end);
		const std::streamsize size = file.tellg();
		file.seekg(0, std::ios::beg);
		std::vector<char> engineData(static_cast<size_t>(size));
		if (!file.read(engineData.data(), size)) {
			std::cerr << "读取引擎失败: " << enginePath << std::endl;
			return false;
		}

		if (cudaSetDevice(0) != cudaSuccess) {
			std::cerr << "设置 CUDA 设备失败" << std::endl;
			return false;
		}

		initLibNvInferPlugins(&gLogger, "");

		runtime_.reset(nvinfer1::createInferRuntime(gLogger));
		if (!runtime_) {
			std::cerr << "创建 TensorRT runtime 失败" << std::endl;
			return false;
		}

		engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
		if (!engine_) {
			std::cerr << "反序列化 TensorRT 引擎失败" << std::endl;
			return false;
		}

		context_.reset(engine_->createExecutionContext());
		if (!context_) {
			std::cerr << "创建执行上下文失败" << std::endl;
			return false;
		}

		const int tensorCount = engine_->getNbIOTensors();
		for (int i = 0; i < tensorCount; ++i) {
			const char* name = engine_->getIOTensorName(i);
			if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
				inputName_ = name;
			} else {
				outputName_ = name;
			}
		}

		if (inputName_.empty() || outputName_.empty()) {
			std::cerr << "未找到输入或输出 tensor" << std::endl;
			return false;
		}

		inputDims_ = engine_->getTensorShape(inputName_.c_str());
		outputDims_ = engine_->getTensorShape(outputName_.c_str());
		inputType_ = engine_->getTensorDataType(inputName_.c_str());
		if (inputDims_.nbDims != 3 || outputDims_.nbDims != 3) {
			std::cerr << "不支持的输入/输出维度" << std::endl;
			return false;
		}

		if (!context_->setInputShape(inputName_.c_str(), inputDims_)) {
			std::cerr << "设置输入维度失败" << std::endl;
			return false;
		}

		outputDims_ = context_->getTensorShape(outputName_.c_str());
		if (outputDims_.nbDims != 3) {
			std::cerr << "获取输出维度失败" << std::endl;
			return false;
		}

		inputCount_ = volume(inputDims_);
		outputCount_ = volume(outputDims_);
		inputBytes_ = inputCount_ * elementSize(inputType_);
		hostInput_.assign(inputBytes_, 0);
		hostOutput_.assign(outputCount_, 0.0f);

		void* inputDevice = nullptr;
		void* outputDevice = nullptr;
		if (cudaStreamCreate(&stream_) != cudaSuccess) {
			std::cerr << "创建 CUDA stream 失败" << std::endl;
			return false;
		}
		if (cudaMalloc(&inputDevice, inputBytes_) != cudaSuccess) {
			std::cerr << "分配输入显存失败" << std::endl;
			return false;
		}
		if (cudaMalloc(&outputDevice, outputCount_ * sizeof(float)) != cudaSuccess) {
			std::cerr << "分配输出显存失败" << std::endl;
			return false;
		}

		inputDevice_.reset(inputDevice);
		outputDevice_.reset(outputDevice);

		std::cout << "输入 Tensor: " << inputName_ << " shape=[" << inputDims_.d[0] << ", " << inputDims_.d[1] << ", " << inputDims_.d[2]
				  << "] type=" << static_cast<int>(inputType_) << " bytes=" << inputBytes_ << std::endl;
		return true;
	}

	std::vector<Detection> infer(const cv::Mat& image, float confThreshold, float iouThreshold) {
		auto start = std::chrono::steady_clock::now();
		const int imageWidth = image.cols;
		const int imageHeight = image.rows;
		const bool hwcInput = inputDims_.d[2] == 3;
		const int targetH = hwcInput ? inputDims_.d[0] : inputDims_.d[1];
		const int targetW = hwcInput ? inputDims_.d[1] : inputDims_.d[2];

		cv::Mat padded;
		if (targetH != targetW) {
			throw std::runtime_error("当前示例仅支持方形输入尺寸");
		}
		const LetterBoxInfo letterBox = letterbox(image, padded, targetH);
		std::cout << "缩放比例: " << letterBox.scale << ", 左边距: " << letterBox.padLeft << ", 上边距: " << letterBox.padTop << std::endl;

		std::cout << "预处理后图片尺寸: " << padded.cols << "x" << padded.rows << std::endl;

		if (inputType_ == nvinfer1::DataType::kUINT8 && hwcInput) {
			cv::Mat contiguous = padded.isContinuous() ? padded : padded.clone();
			if (static_cast<size_t>(contiguous.total() * contiguous.elemSize()) != inputBytes_) {
				throw std::runtime_error("输入尺寸与引擎期望不一致");
			}
			std::memcpy(hostInput_.data(), contiguous.data, inputBytes_);
		} else if (inputType_ == nvinfer1::DataType::kFLOAT && !hwcInput) {
			cv::Mat rgb;
			cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
			rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

			std::vector<cv::Mat> channels(3);
			cv::split(rgb, channels);
			const size_t planeSize = static_cast<size_t>(targetH * targetW);
			std::uint8_t* inputPtr = reinterpret_cast<std::uint8_t*>(hostInput_.data());
			for (int c = 0; c < 3; ++c) {
				std::memcpy(inputPtr + c * planeSize, channels[c].data, planeSize * sizeof(std::uint8_t));
			}
		} else {
			throw std::runtime_error("暂不支持的输入类型或布局，请检查 ONNX 导出与 TensorRT 输入格式");
		}

		auto e1 = std::chrono::steady_clock::now();
		if (cudaMemcpyAsync(inputDevice_.get(), hostInput_.data(), inputBytes_, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
			throw std::runtime_error("输入数据拷贝到 GPU 失败");
		}

		auto e2 = std::chrono::steady_clock::now();
		if (!context_->setTensorAddress(inputName_.c_str(), inputDevice_.get())) {
			throw std::runtime_error("设置输入 tensor 地址失败");
		}
		if (!context_->setTensorAddress(outputName_.c_str(), outputDevice_.get())) {
			throw std::runtime_error("设置输出 tensor 地址失败");
		}

		if (!context_->enqueueV3(stream_)) {
			throw std::runtime_error("TensorRT 推理执行失败");
		}

		auto e3 = std::chrono::steady_clock::now();
		if (cudaMemcpyAsync(hostOutput_.data(), outputDevice_.get(), outputCount_ * sizeof(float), cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
			throw std::runtime_error("输出数据拷贝回 CPU 失败");
		}
		if (cudaStreamSynchronize(stream_) != cudaSuccess) {
			throw std::runtime_error("CUDA stream 同步失败");
		}
		auto e4 = std::chrono::steady_clock::now();
		
		auto res = nms(decode(hostOutput_.data(), imageWidth, imageHeight, letterBox, confThreshold), iouThreshold);
		auto e5 = std::chrono::steady_clock::now();
		std::cout << "推理总耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(e5 - start).count() << " ms;"
			<< "前处理耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(e1 - start).count() << " ms;"
			<< "CPU => GPU 耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(e2 - e1).count() << " ms;"
			<< "推理耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(e3 - e2).count() << " ms;"
			<< "GPU => CPU 耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(e4 - e3).count() << " ms;"
			<< "后处理耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(e5 - e4).count() << " ms" << std::endl;
		return res;
	}

	cv::Mat drawDetections(const cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& labels) const {
		cv::Mat canvas = image.clone();
		for (const auto& det : detections) {
			const cv::Scalar color = colorForClass(det.classId);
			cv::rectangle(canvas, det.box, color, 2);

			const std::string label = (det.classId >= 0 && det.classId < static_cast<int>(labels.size())) ? labels[det.classId] : ("class_" + std::to_string(det.classId));
			std::ostringstream oss;
			oss << label << ' ' << std::fixed << std::setprecision(2) << det.score;
			const std::string text = oss.str();

			int baseline = 0;
			const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
			const int x = std::max(0, static_cast<int>(det.box.x));
			const int y = std::max(0, static_cast<int>(det.box.y) - textSize.height - 6);
			cv::rectangle(canvas, cv::Rect(x, y, textSize.width + 6, textSize.height + 6), color, cv::FILLED);
			cv::putText(canvas, text, cv::Point(x + 3, y + textSize.height + 2), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
		}
		return canvas;
	}

	int classCount() const {
		return featureCount() - 4;
	}

private:
	size_t elementSize(nvinfer1::DataType type) const {
		switch (type) {
		case nvinfer1::DataType::kFLOAT:
			return sizeof(float);
		case nvinfer1::DataType::kHALF:
			return sizeof(std::uint16_t);
		case nvinfer1::DataType::kINT8:
			return sizeof(std::int8_t);
		case nvinfer1::DataType::kINT32:
			return sizeof(std::int32_t);
		case nvinfer1::DataType::kUINT8:
			return sizeof(std::uint8_t);
		case nvinfer1::DataType::kBOOL:
			return sizeof(bool);
		default:
			throw std::runtime_error("未知 TensorRT DataType");
		}
	}

	size_t volume(const nvinfer1::Dims& dims) const {
		size_t total = 1;
		for (int i = 0; i < dims.nbDims; ++i) {
			total *= static_cast<size_t>(dims.d[i]);
		}
		return total;
	}

	int featureCount() const {
		if (outputDims_.d[1] == 6 || outputDims_.d[1] == 84) {
			return outputDims_.d[1];
		}
		return outputDims_.d[2];
	}

	cv::Scalar colorForClass(int classId) const {
		static const std::vector<cv::Scalar> palette = {
			{0, 215, 255},
			{0, 165, 255},
			{0, 255, 127},
			{255, 191, 0},
			{255, 99, 71},
			{255, 0, 255},
		};
		if (classId < 0) {
			return {0, 255, 0};
		}
		return palette[static_cast<size_t>(classId) % palette.size()];
	}

	std::vector<Detection> decode(const float* output, int imageWidth, int imageHeight, const LetterBoxInfo& letterBox, float confThreshold) const {
		const int featureCount = this->featureCount();
		const int numClasses = featureCount - 4;
		const int numPredictions = (outputDims_.d[1] == featureCount) ? outputDims_.d[2] : outputDims_.d[1];
		const bool channelFirst = outputDims_.d[1] == featureCount;

		std::vector<Detection> detections;
		detections.reserve(static_cast<size_t>(numPredictions));

		for (int i = 0; i < numPredictions; ++i) {
			float cx = 0.0f;
			float cy = 0.0f;
			float w = 0.0f;
			float h = 0.0f;
			int classId = -1;
			float score = 0.0f;

			if (channelFirst) {
				cx = output[0 * numPredictions + i];
				cy = output[1 * numPredictions + i];
				w = output[2 * numPredictions + i];
				h = output[3 * numPredictions + i];
				for (int c = 0; c < numClasses; ++c) {
					float current = output[(4 + c) * numPredictions + i];
					if (current > score) {
						score = current;
						classId = c;
					}
				}
			} else {
				const float* pred = output + static_cast<size_t>(i) * featureCount;
				cx = pred[0];
				cy = pred[1];
				w = pred[2];
				h = pred[3];
				for (int c = 0; c < numClasses; ++c) {
					float current = pred[4 + c];
					if (current > score) {
						score = current;
						classId = c;
					}
				}
			}

			if (classId < 0) {
				continue;
			}
			if (score < 0.0f || score > 1.0f) {
				score = sigmoid(score);
			}
			if (score < confThreshold) {
				continue;
			}

			float x1 = cx - 0.5f * w;
			float y1 = cy - 0.5f * h;
			float x2 = cx + 0.5f * w;
			float y2 = cy + 0.5f * h;

			x1 = (x1 - static_cast<float>(letterBox.padLeft)) / letterBox.scale;
			y1 = (y1 - static_cast<float>(letterBox.padTop)) / letterBox.scale;
			x2 = (x2 - static_cast<float>(letterBox.padLeft)) / letterBox.scale;
			y2 = (y2 - static_cast<float>(letterBox.padTop)) / letterBox.scale;

			x1 = std::clamp(x1, 0.0f, static_cast<float>(imageWidth - 1));
			y1 = std::clamp(y1, 0.0f, static_cast<float>(imageHeight - 1));
			x2 = std::clamp(x2, 0.0f, static_cast<float>(imageWidth - 1));
			y2 = std::clamp(y2, 0.0f, static_cast<float>(imageHeight - 1));

			Detection det;
			det.box = cv::Rect2f(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
			det.classId = classId;
			det.score = score;
			detections.push_back(det);
		}

		return detections;
	}

	std::vector<Detection> nms(const std::vector<Detection>& detections, float iouThreshold) const {
		std::vector<Detection> result;
		if (detections.empty()) {
			return result;
		}

		std::vector<int> order(detections.size());
		std::iota(order.begin(), order.end(), 0);
		std::sort(order.begin(), order.end(), [&detections](int lhs, int rhs) {
			return detections[static_cast<size_t>(lhs)].score > detections[static_cast<size_t>(rhs)].score;
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
				if (detections[static_cast<size_t>(current)].classId != detections[static_cast<size_t>(other)].classId) {
					continue;
				}
				if (iou(detections[static_cast<size_t>(current)].box, detections[static_cast<size_t>(other)].box) > iouThreshold) {
					removed[static_cast<size_t>(other)] = true;
				}
			}
		}

		return result;
	}

private:
	std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;
	std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
	std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context_;
	std::unique_ptr<void, CudaDeleter> inputDevice_{nullptr};
	std::unique_ptr<void, CudaDeleter> outputDevice_{nullptr};
	nvinfer1::Dims inputDims_{};
	nvinfer1::Dims outputDims_{};
	nvinfer1::DataType inputType_{nvinfer1::DataType::kFLOAT};
	std::string inputName_;
	std::string outputName_;
	size_t inputCount_ = 0;
	size_t inputBytes_ = 0;
	size_t outputCount_ = 0;
	cudaStream_t stream_ = nullptr;
	std::vector<std::uint8_t> hostInput_;
	std::vector<float> hostOutput_;
};

}  // namespace

int main(int argc, char** argv) {
	const Arguments args = parseArguments(argc, argv);
	if (args.imagePath.empty() && args.imgRootPath.empty()) {
		printUsage(argv[0]);
		return 1;
	}
	YoloTrtDetector detector;
	if (!detector.loadEngine(args.enginePath)) {
		return 1;
	}
	const fs::path imgRoot = !args.imgRootPath.empty() ? fs::path(args.imgRootPath) : fs::path(args.imagePath);
	const std::vector<fs::path> imagePaths = collectImages(imgRoot);
	if (imagePaths.empty()) {
		std::cerr << "未找到可推理的图像: " << imgRoot << std::endl;
		return 1;
	}

	const std::vector<std::string> labels = loadLabels(args.labelsPath, std::max(1, detector.classCount()));
	for (const auto& imagePath : imagePaths) {
		try {
			cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
			if (image.empty()) {
				std::cerr << "无法读取图片: " << imagePath << std::endl;
				continue;
			}

			const auto start = std::chrono::steady_clock::now();
			const std::vector<Detection> detections = detector.infer(image, args.confThreshold, args.iouThreshold);
			const auto end = std::chrono::steady_clock::now();

			cv::Mat rendered = detector.drawDetections(image, detections, labels);

			fs::path outputPath(args.outputPath);
			if (outputPath.has_extension()) {
				outputPath = outputPath.parent_path();
			}
			if (outputPath.empty()) {
				outputPath = "outputs";
			}
			fs::create_directories(outputPath);
			const fs::path outputFile = outputPath / (imagePath.stem().string() + "_result.jpg");
			if (!cv::imwrite(outputFile.string(), rendered)) {
				std::cerr << "保存结果图片失败: " << outputFile << std::endl;
				continue;
			}

			std::chrono::duration<double, std::milli> elapsed = end - start;
			std::cout << "图片: " << imagePath << std::endl;
			std::cout << "推理完成, 耗时: " << elapsed.count() << " ms" << std::endl;
			std::cout << "检测结果数量: " << detections.size() << std::endl;
			for (size_t i = 0; i < detections.size(); ++i) {
				const auto& det = detections[i];
				const std::string label = (det.classId >= 0 && det.classId < static_cast<int>(labels.size())) ? labels[det.classId] : ("class_" + std::to_string(det.classId));
				std::cout << i << ": " << label << " score=" << det.score
						<< " box=[" << det.box.x << ", " << det.box.y << ", " << det.box.width << ", " << det.box.height << "]" << std::endl;
			}
			std::cout << "结果已保存到: " << outputFile << std::endl;
		} catch (const std::exception& e) {
			std::cerr << "执行失败: " << imagePath << " : " << e.what() << std::endl;
		}
	}
}
