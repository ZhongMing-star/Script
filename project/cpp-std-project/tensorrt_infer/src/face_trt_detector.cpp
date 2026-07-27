#include "face_trt_detector.h"

std::vector<VisionCore::FaceDetection2D> FaceTrtDetector::infer(const cv::Mat &inputImage)
{
    if (inputImage.empty())
    {
        throw std::runtime_error("输入图像为空");
    }

    if (m_inputShape[0] != m_inputShape[1])
    {
        throw std::runtime_error("当前实现仅支持方形输入尺寸");
    }

    const VisionCore::LetterBoxInfo letterboxInfo = VisionCore::letterbox(inputImage, m_resizedImage, m_inputShape[0]);

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
        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        const size_t planeSize = static_cast<size_t>(m_inputShape[0] * m_inputShape[1]);
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

    return postprocessInsight(letterboxInfo.scale, inputImage.size());
}

std::vector<VisionCore::FaceDetection2D> FaceTrtDetector::postprocessInsight(float scale,
                                                                             const cv::Size &originalSize) const
{
    std::vector<float> allScores;
    std::vector<float> allBBoxes;
    std::vector<float> allLandmarks;

    for (size_t i = 0; i < m_featStrides.size(); ++i)
    {
        processFeatureLayerInsight(i, scale, originalSize, allScores, allBBoxes, allLandmarks);
    }

    std::vector<VisionCore::FaceDetection2D> candidates;
    candidates.reserve(allScores.size());
    for (size_t i = 0; i < allScores.size(); ++i)
    {
        VisionCore::FaceDetection2D det;
        det.bbox = cv::Rect2f(
            cv::Point2f(allBBoxes[i * 4], allBBoxes[i * 4 + 1]),
            cv::Point2f(allBBoxes[i * 4 + 2], allBBoxes[i * 4 + 3]));
        det.cls_id = 0;
        det.conf = allScores[i];
        if (m_useKps && allLandmarks.size() >= (i + 1) * 10)
        {
            det.landmarks.reserve(5);
            for (int k = 0; k < 5; ++k)
            {
                det.landmarks.emplace_back(allLandmarks[i * 10 + 2 * k], allLandmarks[i * 10 + 2 * k + 1]);
            }
        }
        candidates.push_back(std::move(det));
    }

    return nms(candidates);
}

void FaceTrtDetector::processFeatureLayerInsight(size_t layerIdx,
                                                 float scale,
                                                 const cv::Size &originalSize,
                                                 std::vector<float> &allScores,
                                                 std::vector<float> &allBBoxes,
                                                 std::vector<float> &allLandmarks) const
{
    const int stride = m_featStrides[layerIdx];
    const auto &scoreBinding = m_outputBindings[layerIdx];
    const auto &boxBinding = m_outputBindings[layerIdx + static_cast<size_t>(m_fmc)];
    if (scoreBinding.hostFloat.empty() || boxBinding.hostFloat.empty())
    {
        return;
    }

    const float *scores = scoreBinding.hostFloat.data();
    const float *bboxes = boxBinding.hostFloat.data();

    const int numAnchors = static_cast<int>(scoreBinding.dims.nbDims >= 2
                                                ? std::max(scoreBinding.dims.d[0], scoreBinding.dims.d[1])
                                                : scoreBinding.dims.d[0]);
    const int gridH = m_inputShape[0] / stride;
    const int gridW = m_inputShape[1] / stride;

    const float *landmarks = nullptr;
    if (m_useKps)
    {
        const auto &kpsBinding = m_outputBindings[layerIdx + static_cast<size_t>(m_fmc * 2)];
        if (!kpsBinding.hostFloat.empty())
        {
            landmarks = kpsBinding.hostFloat.data();
        }
    }

    for (int i = 0; i < numAnchors; ++i)
    {
        const float score = scores[i];
        if (score < m_config.conf_threshold)
        {
            continue;
        }

        const float cx = static_cast<float>(((i / 2) % gridW) * stride);
        const float cy = static_cast<float>(((i / 2) / gridH) * stride);

        const float l = bboxes[i * 4] * static_cast<float>(stride);
        const float t = bboxes[i * 4 + 1] * static_cast<float>(stride);
        const float r = bboxes[i * 4 + 2] * static_cast<float>(stride);
        const float b = bboxes[i * 4 + 3] * static_cast<float>(stride);

        float x1 = (cx - l) / scale;
        float y1 = (cy - t) / scale;
        float x2 = (cx + r) / scale;
        float y2 = (cy + b) / scale;

        x1 = std::clamp(x1, 0.0f, static_cast<float>(originalSize.width));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(originalSize.height));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(originalSize.width));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(originalSize.height));

        allScores.push_back(score);
        allBBoxes.insert(allBBoxes.end(), {x1, y1, x2, y2});

        if (landmarks != nullptr)
        {
            for (int k = 0; k < 5; ++k)
            {
                const float lmx = (cx + landmarks[i * 10 + 2 * k] * static_cast<float>(stride)) / scale;
                const float lmy = (cy + landmarks[i * 10 + 2 * k + 1] * static_cast<float>(stride)) / scale;
                allLandmarks.push_back(lmx);
                allLandmarks.push_back(lmy);
            }
        }
    }
}

std::vector<VisionCore::FaceDetection2D> FaceTrtDetector::nms(const std::vector<VisionCore::FaceDetection2D> &detections) const
{
    std::vector<VisionCore::FaceDetection2D> result;
    if (detections.empty())
    {
        return result;
    }

    std::vector<int> order(detections.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&detections](int lhs, int rhs)
              { return detections[static_cast<size_t>(lhs)].conf > detections[static_cast<size_t>(rhs)].conf; });

    std::vector<bool> removed(detections.size(), false);
    for (size_t i = 0; i < order.size(); ++i)
    {
        const int current = order[i];
        if (removed[static_cast<size_t>(current)])
        {
            continue;
        }
        result.push_back(detections[static_cast<size_t>(current)]);
        for (size_t j = i + 1; j < order.size(); ++j)
        {
            const int other = order[j];
            if (removed[static_cast<size_t>(other)])
            {
                continue;
            }
            if (detections[static_cast<size_t>(current)].cls_id != detections[static_cast<size_t>(other)].cls_id)
            {
                continue;
            }
            if (VisionCore::iou(detections[static_cast<size_t>(current)].bbox, detections[static_cast<size_t>(other)].bbox) > m_config.nms_threshold)
            {
                removed[static_cast<size_t>(other)] = true;
            }
        }
    }

    return result;
}

std::string FaceTrtDetector::getConfigStr() const
{
    return "Engine Path: " + m_config.engine_path + "\n" +
           "Confidence Threshold: " + std::to_string(m_config.conf_threshold) + "\n" +
           "NMS Threshold: " + std::to_string(m_config.nms_threshold);
}