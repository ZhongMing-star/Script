import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def collect_images(image_dir: Path, max_images: int) -> List[Path]:
    paths: List[Path] = []
    for root, _, files in os.walk(image_dir):
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(p)
    paths.sort()
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def letterbox_top_left(image: np.ndarray, target_h: int, target_w: int, pad_value: int = 114) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_h / float(h), target_w / float(w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas


class EntropyImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_paths: List[Path],
        input_shape: List[int],
        cache_file: Path,
        input_dtype: np.dtype,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.input_dtype = input_dtype
        self.index = 0

        self.input_size = int(np.prod(self.input_shape))
        self.host_input = np.zeros(self.input_shape, dtype=self.input_dtype)
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)

    def get_batch_size(self) -> int:
        return 1

    def _load_one(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read calibration image: {image_path}")

        if len(self.input_shape) == 3:
            h, w, c = self.input_shape
            if c != 3:
                raise RuntimeError(f"Unsupported input channel count: {c}")
            out = letterbox_top_left(image, h, w, pad_value=114)
            return out.astype(self.input_dtype, copy=False)

        if len(self.input_shape) == 4:
            n, c, h, w = self.input_shape
            if n != 1 or c != 3:
                raise RuntimeError(f"Unsupported NCHW input shape: {self.input_shape}")
            out = letterbox_top_left(image, h, w, pad_value=114)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            out = out.astype(np.float32) / 255.0
            out = np.transpose(out, (2, 0, 1))[None, ...]
            return out.astype(self.input_dtype, copy=False)

        raise RuntimeError(f"Unsupported input rank: {len(self.input_shape)}")

    def get_batch(self, names):
        if self.index >= len(self.image_paths):
            return None

        sample = self._load_one(self.image_paths[self.index])
        self.index += 1
        np.copyto(self.host_input, sample)
        cuda.memcpy_htod(self.device_input, self.host_input)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            print(f"Using existing calibration cache: {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)
        print(f"Calibration cache saved: {self.cache_file}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TensorRT INT8 engine with entropy calibrator")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--workspace-gb", type=int, default=4)
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    image_dir = Path(args.images)
    cache_path = Path(args.cache)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Calibration image directory not found: {image_dir}")

    image_paths = collect_images(image_dir, args.max_images)
    if not image_paths:
        raise RuntimeError("No calibration images found")

    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX")

    if network.num_inputs != 1:
        raise RuntimeError(f"Only single input is supported, got {network.num_inputs}")

    inp = network.get_input(0)
    input_shape = list(inp.shape)
    input_dtype = inp.dtype

    for d in input_shape:
        if d == -1:
            raise RuntimeError(
                "Dynamic input shape is not supported by this script. Please export fixed-shape ONNX first."
            )

    if input_dtype == trt.DataType.UINT8:
        np_dtype = np.uint8
    elif input_dtype == trt.DataType.FLOAT:
        np_dtype = np.float32
    else:
        raise RuntimeError(f"Unsupported input dtype for calibrator script: {input_dtype}")

    print(f"Calibration images: {len(image_paths)}")
    print(f"Network input shape: {input_shape}")
    print(f"Network input dtype: {input_dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30))

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)

    calibrator = EntropyImageCalibrator(
        image_paths=image_paths,
        input_shape=input_shape,
        cache_file=cache_path,
        input_dtype=np_dtype,
    )
    config.int8_calibrator = calibrator

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized TensorRT engine")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)
    print(f"INT8 engine saved to: {engine_path}")


if __name__ == "__main__":
    main()
