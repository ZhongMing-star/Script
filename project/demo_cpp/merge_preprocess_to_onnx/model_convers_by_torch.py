import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import onnx
import onnxruntime as ort


class PreprocessModule(nn.Module):
    """PyTorch实现图像预处理模块（对应原ONNX手动节点逻辑）"""
    def __init__(self, target_size=640, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), bgr2rgb=True):
        super().__init__()
        self.target_size = target_size
        self.mean = torch.tensor(mean).reshape(1, 3, 1, 1)
        self.std = torch.tensor(std).reshape(1, 3, 1, 1)
        self.bgr2rgb = bgr2rgb

    def forward(self, raw_image: torch.Tensor):
        """
        输入: raw_image - [H, W, 3] uint8 原始图像
        输出: 预处理后的张量 [1, 3, target_size, target_size] float32
        """
        # 1. UINT8 -> FLOAT32 + BGR2RGB（如需）
        x = raw_image.to(torch.float32)
        if self.bgr2rgb:
            x = x[..., [2, 1, 0]]  # HWC BGR -> RGB
        
        # 2. HWC -> CHW
        x = x.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        # 3. 等比例缩放（保持宽高比）
        h, w = x.shape[2], x.shape[3]
        scale = min(self.target_size / h, self.target_size / w)
        new_h = int((h * scale + 0.5) // 1)  # 等价原逻辑的floor(h*scale + 0.5)
        new_w = int((w * scale + 0.5) // 1)
        x = F.resize(x, [new_h, new_w], interpolation=F.InterpolationMode.BILINEAR)
        
        # 4. 填充到目标尺寸（黑边）
        pad_bottom = self.target_size - new_h
        pad_right = self.target_size - new_w
        x = F.pad(x, [0, pad_right, 0, pad_bottom], fill=0.0)
        
        # 5. 归一化 (x/255 - mean) / std
        x = x / 255.0
        return x


def export_onnx_with_preprocess(
    input_onnx_path: str,
    output_onnx_path: str,
    target_size: int = 640,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    bgr2rgb: bool = True
):
    """
    步骤1: 导出预处理模块为ONNX
    步骤2: 合并原始ONNX模型（将预处理ONNX与原模型拼接）
    """
    # ---------------------- 1. 导出预处理模块为ONNX ----------------------
    preprocess_model = PreprocessModule(target_size, mean, std, bgr2rgb)
    preprocess_model.eval()
    
    # 构造示例输入（动态尺寸）
    dummy_input = torch.randint(0, 255, (1080, 1920, 3), dtype=torch.uint8)
    dynamic_axes = {
        "raw_image": {0: "H", 1: "W"},  # 输入动态尺寸
        "preprocess_output": {2: "target_h", 3: "target_w"}  # 输出固定为target_size，可省略
    }
    
    # 预留临时文件名，实际导出将在读取原始模型后使用匹配的 opset
    preprocess_onnx_path = "temp_preprocess.onnx"
    
    # ---------------------- 2. 合并预处理ONNX与原始模型 ----------------------
    # 加载原始ONNX（先加载原模型以读取其 opset 版本）
    original_onnx = onnx.load(input_onnx_path)

    # 尝试将原始模型转换为一个较新的 opset（18），以便与 PyTorch 导出的子图兼容
    desired_opset = 13
    try:
        orig_opset = original_onnx.opset_import[0].version
    except Exception:
        orig_opset = None

    if orig_opset != desired_opset:
        try:
            original_onnx = onnx.version_converter.convert_version(original_onnx, desired_opset)
            print(f"原始模型已转换到 opset {desired_opset}")
        except Exception as e:
            print(f"警告: 无法将原始模型转换到 opset {desired_opset}: {e}")

    # 使用目标 opset 导出预处理子图
    torch.onnx.export(
        preprocess_model,
        dummy_input,
        preprocess_onnx_path,
        input_names=["raw_image"],
        output_names=["preprocess_output"],
        dynamic_axes=dynamic_axes,
        opset_version=desired_opset,
        do_constant_folding=True
    )

    # 加载预处理ONNX
    preprocess_onnx = onnx.load(preprocess_onnx_path)
    
    # 修正原始模型输入：将原始输入替换为预处理输出
    for node in original_onnx.graph.node:
        for i, inp in enumerate(node.input):
            if inp == original_onnx.graph.input[0].name:
                node.input[i] = "preprocess_output"

    # 合并图：预处理节点 + 原始模型节点
    merged_graph = onnx.helper.make_graph(
        nodes=list(preprocess_onnx.graph.node) + list(original_onnx.graph.node),
        name="merged_graph",
        inputs=list(preprocess_onnx.graph.input),  # 输入改为raw_image
        outputs=list(original_onnx.graph.output),
        initializer=list(preprocess_onnx.graph.initializer) + list(original_onnx.graph.initializer)
    )
    
    # 构建合并后的ONNX模型
    merged_model = onnx.helper.make_model(
        merged_graph,
        producer_name="pytorch-export",
        opset_imports=preprocess_onnx.opset_import
    )
    
    # 检查并保存
    onnx.checker.check_model(merged_model)
    onnx.save(merged_model, output_onnx_path)
    print(f"✅ 合并后的模型已保存: {output_onnx_path}")
    
    # 清理临时文件
    import os
    os.remove(preprocess_onnx_path)
    
    # 尝试生成兼容较旧 onnxruntime 的 IR11 版本文件（非保证成功）
    try:
        ir11_path = output_onnx_path.replace('.onnx', '_ir11.onnx')
        try:
            converted = onnx.version_converter.convert_version(merged_model, 11)
            onnx.save(converted, ir11_path)
            print(f"尝试将合并模型转换到 opset/IR 11 并保存为: {ir11_path}")
        except Exception:
            # 直接尝试降级 ir_version（有风险，可能不兼容）
            merged_model.ir_version = 11
            onnx.save(merged_model, ir11_path)
            print(f"已写入降级 ir_version=11 的模型到: {ir11_path} （可能不兼容）")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="用PyTorch导出方式实现ONNX预处理节点插入")
    parser.add_argument("--input", help="原始ONNX模型路径")
    parser.add_argument("--output", help="输出ONNX模型路径")
    parser.add_argument("--size", type=int, default=640, help="目标尺寸，默认640")
    parser.add_argument(
        "--no-bgr2rgb",
        dest="bgr2rgb",
        action="store_false",
        help="禁用BGR->RGB转换"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 导出并合并模型
    export_onnx_with_preprocess(
        input_onnx_path=args.input,
        output_onnx_path=args.output,
        target_size=args.size,
        bgr2rgb=args.bgr2rgb
    )
    
    # 推理自测
    try:
        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        # 测试多种尺寸
        for hh, ww in [(512, 720), (800, 400), (640, 640)]:
            test_img = np.random.randint(0, 255, (hh, ww, 3), dtype=np.uint8)
            sess.run(None, {"raw_image": test_img})
        print("✅ 多尺寸图像推理测试全部通过")
    except Exception as e:
        print(f"❌ 推理异常: {e}")
        import traceback
        traceback.print_exc()
        # 如果存在 IR11 备份文件，尝试加载它
        import os
        ir11_path = args.output.replace('.onnx', '_ir11.onnx')
        if os.path.exists(ir11_path):
            print(f"尝试加载 IR11 备份模型: {ir11_path}")
            try:
                sess = ort.InferenceSession(ir11_path, providers=["CPUExecutionProvider"])
                for hh, ww in [(512, 720), (800, 400), (640, 640)]:
                    test_img = np.random.randint(0, 255, (hh, ww, 3), dtype=np.uint8)
                    sess.run(None, {"raw_image": test_img})
                print("✅ IR11 备份模型推理测试全部通过")
            except Exception as e2:
                print(f"❌ IR11 模型加载/推理也失败: {e2}")
                print("建议升级 onnxruntime 到支持更高 IR/opset 的版本，例如：")
                print("pip install --upgrade onnxruntime")
                print("或 (GPU): pip install --upgrade onnxruntime-gpu")
        else:
            print("未找到 IR11 备份模型。建议升级 onnxruntime：")
            print("pip install --upgrade onnxruntime")
            print("或 (GPU): pip install --upgrade onnxruntime-gpu")