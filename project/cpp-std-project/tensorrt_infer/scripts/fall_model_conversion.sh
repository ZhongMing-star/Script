
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/resource/models/det_500m.onnx}"
MODEL_BASENAME="$(basename "${MODEL_PATH}")"
MODEL_STEM="${MODEL_BASENAME%.*}"

ONNX_DIR="${ONNX_DIR:-${ROOT_DIR}/resource/models}"
ENGINE_DIR="${ENGINE_DIR:-${ROOT_DIR}/resource/models}"
CALIB_DIR="${CALIB_DIR:-${ROOT_DIR}/resource/images}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRTEXEC_BIN="${TRTEXEC_BIN:-}"
MAX_IMAGES="${MAX_IMAGES:-0}"
WORKSPACE_GB="${WORKSPACE_GB:-4}"
SIZES="${SIZES:-640 1280}"
CLEAN_INTERMEDIATE="${CLEAN_INTERMEDIATE:-1}"

EXPORT_SCRIPT="${ROOT_DIR}/scripts/InspireFace_model_conversion/01_export_end2end_InspireFace_onnx.py"
CALIB_SCRIPT="${ROOT_DIR}/scripts/InspireFace_model_conversion/02_build_int8_engine_with_calibrator.py"

if [[ -z "${TRTEXEC_BIN}" ]]; then
  if [[ -x "/usr/local/TensorRT-10.14.1.48/bin/trtexec" ]]; then
    TRTEXEC_BIN="/usr/local/TensorRT-10.14.1.48/bin/trtexec"
  else
    TRTEXEC_BIN="trtexec"
  fi
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ONNX model not found: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${EXPORT_SCRIPT}" ]]; then
  echo "Export script not found: ${EXPORT_SCRIPT}"
  exit 1
fi

if [[ ! -f "${CALIB_SCRIPT}" ]]; then
  echo "Calibrator script not found: ${CALIB_SCRIPT}"
  exit 1
fi

if [[ ! -d "${CALIB_DIR}" ]]; then
  echo "Calibration image directory not found: ${CALIB_DIR}"
  exit 1
fi

mkdir -p "${ONNX_DIR}" "${ENGINE_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Sizes: ${SIZES}"
echo "ONNX output dir: ${ONNX_DIR}"
echo "Engine output dir: ${ENGINE_DIR}"
echo "Calibration image dir: ${CALIB_DIR}"
echo "trtexec: ${TRTEXEC_BIN}"
echo "python: ${PYTHON_BIN}"
echo "clean intermediate: ${CLEAN_INTERMEDIATE}"
echo ""

for SIZE in ${SIZES}; do
  ONNX_PATH="${ONNX_DIR}/${MODEL_STEM}_end2end_static_${SIZE}.onnx"

  FP16_ENGINE="${ENGINE_DIR}/${MODEL_STEM}_end2end_fp16_${SIZE}.trt"
  INT8_ENGINE="${ENGINE_DIR}/${MODEL_STEM}_end2end_int8_${SIZE}.trt"
  CALIB_CACHE="${ENGINE_DIR}/${MODEL_STEM}_end2end_int8_${SIZE}.calib"

  echo "========== [${SIZE}x${SIZE}] =========="

  if [[ -f "${ONNX_PATH}" ]]; then
    echo "[1/3] ONNX already exists, skip: ${ONNX_PATH}"
  else
    echo "[1/3] Exporting static ONNX..."
    "${PYTHON_BIN}" "${EXPORT_SCRIPT}" \
      --onnx "${MODEL_PATH}" \
      --input_size "${SIZE}" "${SIZE}" \
      --save_onnx "${ONNX_PATH}"
  fi

  if [[ -f "${FP16_ENGINE}" ]]; then
    echo "[2/3] FP16 engine already exists, skip: ${FP16_ENGINE}"
  else
    echo "[2/3] Building FP16 engine..."
    "${TRTEXEC_BIN}" \
      --onnx="${ONNX_PATH}" \
      --saveEngine="${FP16_ENGINE}" \
      --fp16
  fi

  if [[ -f "${CALIB_CACHE}" ]]; then
    echo "[3/3] INT8 calibration cache already exists, skip: ${CALIB_CACHE}"
  else
    echo "[3/3] Generating INT8 calibration cache..."
    "${PYTHON_BIN}" "${CALIB_SCRIPT}" \
      --onnx "${ONNX_PATH}" \
      --engine "${INT8_ENGINE}" \
      --images "${CALIB_DIR}" \
      --cache "${CALIB_CACHE}" \
      --max-images "${MAX_IMAGES}" \
      --workspace-gb "${WORKSPACE_GB}"
  fi

  echo "[done] ONNX: ${ONNX_PATH}"
  echo "[done] FP16: ${FP16_ENGINE}"
  echo "[done] INT8: ${INT8_ENGINE}"
  echo "[done] CALIB: ${CALIB_CACHE}"
  echo ""
done

# Cleanup intermediate artifacts if requested

if [[ "${CLEAN_INTERMEDIATE}" == "1" ]]; then
  echo "Cleaning intermediate artifacts (.onnx/.onnx.data/.calib/*_calib_tmp_*.trt)..."

  if [[ -d "${ONNX_DIR}" ]]; then
    find "${ONNX_DIR}" -maxdepth 1 -type f \( -name '*.onnx' -o -name '*.onnx.data' \) -print -delete
  fi

  if [[ -d "${ENGINE_DIR}" ]]; then
    find "${ENGINE_DIR}" -maxdepth 1 -type f \( -name '*.calib' -o -name '*_calib_tmp_*.trt' \) -print -delete
  fi

  echo "Cleanup finished. Preserved original PT model and final TRT engines."
fi

echo "All conversions completed."
