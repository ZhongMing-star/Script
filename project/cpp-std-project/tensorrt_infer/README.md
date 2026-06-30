# YOLO 模型转换脚本说明

本文档说明如何使用 [yolo_model_conversion.sh](scripts/yolo_model_conversion.sh) 将 YOLO `.pt` 模型批量转换为 TensorRT 推理产物。

脚本会按输入尺寸循环执行以下流程：

1. 导出静态 ONNX
2. 生成 FP16 TensorRT 引擎
3. 生成 INT8 校准缓存（calib）
4. 基于校准缓存生成 INT8 TensorRT 引擎

可选地，流程结束后会清理中间文件（ONNX、校准缓存、临时校准引擎）。

## 1. 脚本位置

- 主脚本：[scripts/yolo_model_conversion.sh](scripts/yolo_model_conversion.sh)

## 2. 前置依赖

执行前请确认以下环境可用：

- Python 3（默认命令：`python3`）
- TensorRT `trtexec`（默认查找 `/usr/local/TensorRT-10.14.1.48/bin/trtexec`，找不到则回退到 PATH 中的 `trtexec`）
- 模型导出脚本与校准脚本（由主脚本内部调用）
- 校准图片目录存在且可读

## 3. 默认路径与参数

以下为脚本中可通过环境变量覆盖的默认值：

- `MODEL_PATH`：`${ROOT_DIR}/resource/models/fall_detection.pt`
- `ONNX_DIR`：`${ROOT_DIR}/resource/models`
- `ENGINE_DIR`：`${ROOT_DIR}/resource/models`
- `CALIB_DIR`：`${ROOT_DIR}/resource/images`
- `PYTHON_BIN`：`python3`
- `TRTEXEC_BIN`：自动探测（见上）
- `MAX_IMAGES`：`0`（表示不限制）
- `WORKSPACE_GB`：`4`
- `SIZES`：`640 1280`
- `CLEAN_INTERMEDIATE`：`1`（启用清理）

## 4. 快速开始

在 [scripts](scripts) 目录执行：

```bash
cd scripts
chmod +x yolo_model_conversion.sh
./yolo_model_conversion.sh
```

## 5. 常用执行示例

### 5.1 指定模型与尺寸

```bash
MODEL_PATH="/path/to/model.pt" \
SIZES="640 960 1280" \
./scripts/yolo_model_conversion.sh
```

### 5.2 指定 Python 与 trtexec

```bash
PYTHON_BIN="python3.10" \
TRTEXEC_BIN="/usr/local/TensorRT-10.14.1.48/bin/trtexec" \
./scripts/yolo_model_conversion.sh
```

### 5.3 控制校准参数

```bash
CALIB_DIR="/path/to/calib_images" \
MAX_IMAGES=500 \
WORKSPACE_GB=8 \
./scripts/yolo_model_conversion.sh
```

### 5.4 保留中间文件（不清理）

```bash
CLEAN_INTERMEDIATE=0 ./scripts/yolo_model_conversion.sh
```

## 6. 产物命名规则

设模型文件名为 `xxx.pt`，尺寸为 `640`，则产物类似：

- ONNX：`xxx_end2end_static_640.onnx`
- FP16 引擎：`xxx_end2end_fp16_640.trt`
- INT8 校准缓存：`xxx_end2end_int8_640.calib`
- INT8 引擎：`xxx_end2end_int8_640.trt`
- 临时校准引擎：`xxx_end2end_int8_calib_tmp_640.trt`

## 7. 清理逻辑说明

当 `CLEAN_INTERMEDIATE=1` 时，脚本会删除：

- `ONNX_DIR` 下的 `*.onnx` 与 `*.onnx.data`
- `ENGINE_DIR` 下的 `*.calib` 与 `*_calib_tmp_*.trt`

脚本会保留：

- 原始 `.pt` 模型
- 最终 `.trt` 引擎（FP16 与 INT8）

## 8. 常见问题排查

- 报错 `PT model not found`：检查 `MODEL_PATH` 是否正确。
- 报错 `Calibration image directory not found`：检查 `CALIB_DIR` 是否存在。
- 报错找不到 `trtexec`：设置 `TRTEXEC_BIN` 为 TensorRT 的完整路径。
- 导出脚本找不到：确认仓库中对应 Python 脚本路径与主脚本配置一致。

## 9. 返回码

脚本启用了 `set -euo pipefail`：

- 任一步骤失败会立即退出
- 退出码非 0 表示执行失败

## 10. InspireFace 脚本补充说明

除 [scripts/yolo_model_conversion.sh](scripts/yolo_model_conversion.sh) 外，仓库还提供了 InspireFace 转换脚本：

- 主脚本：[scripts/InspireFace_model_conversion.sh](scripts/InspireFace_model_conversion.sh)

该脚本整体流程与 YOLO 转换脚本一致：

1. 导出静态 ONNX
2. 构建 FP16 TensorRT 引擎
3. 生成 INT8 校准缓存
4. 基于校准缓存构建 INT8 TensorRT 引擎

### 10.1 快速开始

```bash
cd scripts
chmod +x InspireFace_model_conversion.sh
./InspireFace_model_conversion.sh
```

### 10.2 常用参数

`InspireFace_model_conversion.sh` 支持通过环境变量覆盖默认配置：

- `MODEL_PATH`：默认 `${ROOT_DIR}/resource/models/fall_detection.pt`
- `ONNX_DIR`：默认 `${ROOT_DIR}/resource/models`
- `ENGINE_DIR`：默认 `${ROOT_DIR}/resource/models`
- `CALIB_DIR`：默认 `${ROOT_DIR}/resource/images`
- `PYTHON_BIN`：默认 `python3`
- `TRTEXEC_BIN`：自动探测 TensorRT `trtexec`，否则回退 PATH
- `MAX_IMAGES`：默认 `64`
- `WORKSPACE_GB`：默认 `4`
- `SIZES`：默认 `640 1280`
- `CLEAN_INTERMEDIATE`：默认 `1`

### 10.3 执行示例

```bash
MODEL_PATH="/path/to/inspireface_model.pt" \
SIZES="640 1280" \
./scripts/InspireFace_model_conversion.sh
```

```bash
PYTHON_BIN="python3.10" \
TRTEXEC_BIN="/usr/local/TensorRT-10.14.1.48/bin/trtexec" \
MAX_IMAGES=500 \
WORKSPACE_GB=8 \
./scripts/InspireFace_model_conversion.sh
```

```bash
CLEAN_INTERMEDIATE=0 ./scripts/InspireFace_model_conversion.sh
```

### 10.4 清理行为说明

当 `CLEAN_INTERMEDIATE=1` 时，脚本会清理：

- `ONNX_DIR` 下的 `*.onnx` 与 `*.onnx.data`
- `ENGINE_DIR` 下的 `*.calib` 与 `*_calib_tmp_*.trt`

同时会保留：

- 原始 `.pt` 模型
- 最终 `.trt` 引擎
- `det_500m.onnx`（清理 ONNX 时显式排除）
