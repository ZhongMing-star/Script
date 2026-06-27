import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx.compose import merge_models


class PreprocessModule(nn.Module):
    """
    TensorRT 兼容预处理模块（letterbox resize + normalize）
    输入 raw_image: [1, 3, H, W] float32 NCHW BGR (OpenCV 原图转 float)
    输出: [1, 3, target_size, target_size] float32 NCHW RGB 归一化图像

    所有尺寸计算均使用张量运算，确保 ONNX 导出时可常量折叠，
    避免 TRT 不支持的动态 shape 节点。
    """
    def __init__(
        self,
        target_size=640,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.target_size = target_size
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1))

    def forward(self, raw_image: torch.Tensor):
        # 输入: [H, W, C] (OpenCV BGR float32 格式)
        # 转为: [1, C, H, W] (NCHW RGB 格式)
        
        # [H, W, C] -> [C, H, W]
        x = raw_image.permute(2, 0, 1)
        # [C, H, W] -> [1, C, H, W]
        x = x.unsqueeze(0)
        
        _, c, h, w = x.shape

        target = torch.tensor([self.target_size], dtype=torch.float32, device=x.device)
        h_f = torch.tensor([h], dtype=torch.float32, device=x.device)
        w_f = torch.tensor([w], dtype=torch.float32, device=x.device)

        # 等比例缩放因子
        scale = torch.minimum(target / h_f, target / w_f)

        # letterbox 新尺寸（张量流，导出时为常量节点）
        new_h_t = torch.floor(h_f * scale + 0.5)
        new_w_t = torch.floor(w_f * scale + 0.5)

        # 转为 Python int（ONNX 常量折叠后成为固定值，TRT 可处理）
        new_h = int(new_h_t.item())
        new_w = int(new_w_t.item())

        # 双线性插值 — size 只传空间维度 (H, W)，opset 17 标准 Resize 算子
        x = F.interpolate(
            x,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

        # 右下补零至 target_size x target_size（Python int -> ONNX 常量）
        pad_bottom = self.target_size - new_h
        pad_right = self.target_size - new_w
        x = F.pad(x, [0, pad_right, 0, pad_bottom], value=0.0)

        # 归一化：/255 -> (x - mean) / std
        x = x / 255.0
        # 分别减均值和除方差，避免复杂融合
        x = x - self.mean
        x = x / self.std
        return x


def export_onnx_with_preprocess(
    input_onnx_path: str,
    output_onnx_path: str,
    target_size: int = 640,
    input_h: int = 1080,
    input_w: int = 1920,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    # 1. 初始化预处理模型
    preprocess_model = PreprocessModule(target_size, mean, std)
    preprocess_model.eval()

    # 固定形状 float32 输入（HWC 格式，模拟 OpenCV 读取的图像）
    dummy_input = torch.randn(input_h, input_w, 3, dtype=torch.float32)
    dynamic_axes = {
        "raw_image": {0: "H", 1: "W"},
        "preprocess_output": {}
    }
    temp_prep_raw = "temp_prep_raw.onnx"
    temp_prep_slim = "temp_prep_slim.onnx"
    opset_version = 18

    # 2. 导出预处理子图（固定形状，无 dynamic_axes）
    torch.onnx.export(
        preprocess_model,
        dummy_input,
        temp_prep_raw,
        input_names=["raw_image"],
        output_names=["preprocess_output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # 3. onnxsim 简化，消除冗余节点
    os.system(f"python -m onnxsim {temp_prep_raw} {temp_prep_slim}")

    # # 4. 校验预处理子图
    # prep_model = onnx.load(temp_prep_slim)
    # onnx.checker.check_model(prep_model)
    # print("preprocess subgraph check passed")

    # # 5. 加载主干模型并拼接
    # main_model = onnx.load(input_onnx_path)
    # combined_model = merge_models(
    #     prep_model,
    #     main_model,
    #     io_map=[("preprocess_output", main_model.graph.input[0].name)],
    # )

    # # 6. 保存并全局简化
    # onnx.save(combined_model, output_onnx_path)
    # os.system(f"python -m onnxsim {output_onnx_path} {output_onnx_path}")

    # # 7. 最终校验
    # final_model = onnx.load(output_onnx_path)
    # onnx.checker.check_model(final_model)
    # print(f"combined model exported: {output_onnx_path}")

    # # 清理临时文件
    # for f in [temp_prep_raw, temp_prep_slim]:
    #     if os.path.exists(f):
    #         os.remove(f)


def parse_args():
    parser = argparse.ArgumentParser(description="preprocess-embedded ONNX export (TRT compatible)")
    parser.add_argument("--input", required=False, help="original backbone ONNX model path")
    parser.add_argument("--output", required=False, help="output combined ONNX path with preprocess")
    parser.add_argument("--size", type=int, default=640, help="model input target size")
    parser.add_argument("--input-h", type=int, default=1080, help="raw input image height (fixed shape)")
    parser.add_argument("--input-w", type=int, default=1920, help="raw input image width (fixed shape)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        # export preprocess + backbone combined ONNX
        export_onnx_with_preprocess(
            input_onnx_path=args.input,
            output_onnx_path=args.output,
            target_size=args.size,
            input_h=args.input_h,
            input_w=args.input_w,
        )

        # # ONNX Runtime inference self-test (4D NCHW float32, fixed shape)
        # import onnxruntime as ort

        # print("\n===== inference self-test =====")
        # sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        # test_img = np.random.randn(1, 3, args.input_h, args.input_w).astype(np.float32) * 128.0
        # out = sess.run(None, {"raw_image": test_img})
        # print(f"  input shape [1,3,{args.input_h},{args.input_w}] -> output shape: {out[0].shape}")
        # print("\nself-test passed (fixed shape; for other resolutions re-export with --input-h/--input-w)")

        # # print trtexec conversion command
        # trt_output = args.output.replace(".onnx", ".engine")
        # trtexec_cmd = (
        #     f"trtexec "
        #     f"--onnx={args.output} "
        #     f"--saveEngine={trt_output} "
        #     f"--fp16 "
        #     f"--shapes=raw_image:[1,3,{args.input_h},{args.input_w}]"
        # )
        # print(f"\n===== trtexec conversion command =====")
        # print(trtexec_cmd)

    except Exception as e:
        print(f"\nexecution failed: {e}")
        import traceback
        traceback.print_exc()


#  trtexec --onnx=temp_prep_slim.onnx --saveEngine=temp_prep_slim.trt --minShapes=raw_image:640x640x3 --optShapes=raw_image:640x640x3 --maxShapes=raw_image:640x640x3
