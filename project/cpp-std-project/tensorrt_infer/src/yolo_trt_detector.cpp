#include "yolo_trt_detector.h"

int YoloTrtDetector::featureCountFromDims(const nvinfer1::Dims &dims, int numClasses)
{
    if (dims.nbDims != 3)
    {
        throw std::runtime_error("不支持的输出维度");
    }

    const int expectedFeatureCount = numClasses + 4;
    if (dims.d[1] == expectedFeatureCount || dims.d[1] == 6 || dims.d[1] == 84)
    {
        return dims.d[1];
    }
    if (dims.d[2] == expectedFeatureCount || dims.d[2] == 6 || dims.d[2] == 84)
    {
        return dims.d[2];
    }
    return dims.d[1];
}

std::string YoloTrtDetector::getConfigStr() const
{
    std::ostringstream oss;
    oss << "Engine Path: " << m_config.engine_path << "\n"
        << "Num Classes: " << m_config.num_classes << "\n"
        << "Confidence Threshold: " << m_config.conf_threshold << "\n"
        << "NMS Threshold: " << m_config.nms_threshold;
    return oss.str();
}

std::vector<VisionCore::Detection2D> YoloTrtDetector::infer(const cv::Mat &inputImage)
{
    if (inputImage.empty())
    {
        throw std::runtime_error("输入图像为空");
    }

    const bool hwcInput = m_inputDims.nbDims == 3 && m_inputDims.d[2] == 3;
    const int targetH = hwcInput ? m_inputDims.d[0] : m_inputDims.d[1];
    const int targetW = hwcInput ? m_inputDims.d[1] : m_inputDims.d[2];
    if (targetH != targetW)
    {
        throw std::runtime_error("当前实现仅支持方形输入尺寸");
    }

    const VisionCore::LetterBoxInfo letterboxInfo = VisionCore::letterbox(inputImage, m_resizedImage, targetH);

    if (m_inputType == nvinfer1::DataType::kUINT8 && hwcInput)
    {
        cv::Mat contiguous = m_resizedImage.isContinuous() ? m_resizedImage : m_resizedImage.clone();
        if (static_cast<size_t>(contiguous.total() * contiguous.elemSize()) != m_inputBytes)
        {
            throw std::runtime_error("输入尺寸与引擎期望不一致");
        }
        std::memcpy(m_hostInput.data(), contiguous.data, m_inputBytes);
    }
    else if (m_inputType == nvinfer1::DataType::kFLOAT && !hwcInput)
    {
        cv::Mat rgb;
        cv::cvtColor(m_resizedImage, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        const size_t planeSize = static_cast<size_t>(targetH * targetW);
        float *inputPtr = reinterpret_cast<float *>(m_hostInput.data());
        for (int c = 0; c < 3; ++c)
        {
            std::memcpy(inputPtr + c * planeSize, channels[c].data, planeSize * sizeof(float));
        }
    }
    else
    {
        throw std::runtime_error("暂不支持的输入类型或布局，请检查引擎输入格式");
    }

    this->_infer();

    return VisionCore::nms(decode(m_outputBindings[0].hostFloat.data(), letterboxInfo), m_config.nms_threshold);
}

std::vector<VisionCore::Detection2D> YoloTrtDetector::decode(const float *output, const VisionCore::LetterBoxInfo &letterboxInfo)
{
    const int featureCount = featureCountFromDims(m_outputDims, m_config.num_classes);
    const int numClasses = featureCount - 4;
    const int numPredictions = (m_outputDims.d[1] == featureCount) ? m_outputDims.d[2] : m_outputDims.d[1];
    const bool channelFirst = m_outputDims.d[1] == featureCount;

    std::vector<VisionCore::Detection2D> detections;
    detections.reserve(static_cast<size_t>(numPredictions));

    for (int i = 0; i < numPredictions; ++i)
    {
        float cx = 0.0f;
        float cy = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
        int clsId = -1;
        float score = 0.0f;

        if (channelFirst)
        {
            cx = output[0 * numPredictions + i];
            cy = output[1 * numPredictions + i];
            w = output[2 * numPredictions + i];
            h = output[3 * numPredictions + i];
            for (int c = 0; c < numClasses; ++c)
            {
                const float current = output[(4 + c) * numPredictions + i];
                if (current > score)
                {
                    score = current;
                    clsId = c;
                }
            }
        }
        else
        {
            const float *pred = output + static_cast<size_t>(i) * featureCount;
            cx = pred[0];
            cy = pred[1];
            w = pred[2];
            h = pred[3];
            for (int c = 0; c < numClasses; ++c)
            {
                const float current = pred[4 + c];
                if (current > score)
                {
                    score = current;
                    clsId = c;
                }
            }
        }

        if (clsId < 0)
        {
            continue;
        }
        if (score < 0.0f || score > 1.0f)
        {
            score = VisionCore::sigmoid(score);
        }
        if (score < m_config.conf_threshold)
        {
            continue;
        }

        float x1 = cx - 0.5f * w;
        float y1 = cy - 0.5f * h;
        float x2 = cx + 0.5f * w;
        float y2 = cy + 0.5f * h;

        x1 = (x1 - static_cast<float>(letterboxInfo.padLeft)) / letterboxInfo.scale;
        y1 = (y1 - static_cast<float>(letterboxInfo.padTop)) / letterboxInfo.scale;
        x2 = (x2 - static_cast<float>(letterboxInfo.padLeft)) / letterboxInfo.scale;
        y2 = (y2 - static_cast<float>(letterboxInfo.padTop)) / letterboxInfo.scale;

        x1 = std::clamp(x1, 0.0f, static_cast<float>(letterboxInfo.originalWidth - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(letterboxInfo.originalHeight - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(letterboxInfo.originalWidth - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(letterboxInfo.originalHeight - 1));

        VisionCore::Detection2D detection;
        detection.bbox = cv::Rect2f(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
        detection.cls_id = clsId;
        detection.conf = score;
        detections.push_back(detection);
    }

    return detections;
}
