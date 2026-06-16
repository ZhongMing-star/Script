import argparse

import onnx
from onnx import helper, TensorProto

import onnxruntime as ort
import numpy as np

def insert_preprocess_nodes(
    model: onnx.ModelProto,
    target_size: int = 640,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    bgr2rgb: bool = True,
) -> onnx.ModelProto:
    graph = model.graph
    orig_input = graph.input[0]
    orig_input_name = orig_input.name

    new_input_name = "raw_image"
    new_input = helper.make_tensor_value_info(
        new_input_name,
        TensorProto.UINT8,
        ["H", "W", 3],
    )
    graph.input[0].CopyFrom(new_input)

    node_list = []
    inter = new_input_name

    # 1. UINT8 -> FLOAT32
    node_list.append(
        helper.make_node(
            "Cast",
            inputs=[inter],
            outputs=["cast_float"],
            to=TensorProto.FLOAT,
        )
    )
    inter = "cast_float"

    # 2. BGR -> RGB
    if bgr2rgb:
        graph.initializer.append(
            helper.make_tensor(
                "bgr2rgb_idx",
                TensorProto.INT64,
                [3],
                [2, 1, 0],
            )
        )
        node_list.append(
            helper.make_node(
                "Gather",
                inputs=[inter, "bgr2rgb_idx"],
                outputs=["rgb_float"],
                axis=2,
            )
        )
        inter = "rgb_float"

    # 3. 计算等比例缩放尺寸
    node_list.append(helper.make_node("Shape", [inter], ["raw_shape"]))
    graph.initializer.extend(
        [
            helper.make_tensor("idx_h", TensorProto.INT64, [1], [0]),
            helper.make_tensor("idx_w", TensorProto.INT64, [1], [1]),
        ]
    )
    node_list.extend(
        [
            helper.make_node("Gather", ["raw_shape", "idx_h"], ["h_dim"], axis=0),
            helper.make_node("Gather", ["raw_shape", "idx_w"], ["w_dim"], axis=0),
            helper.make_node("Cast", ["h_dim"], ["h_float"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["w_dim"], ["w_float"], to=TensorProto.FLOAT),
        ]
    )

    graph.initializer.append(
        helper.make_tensor("size_const", TensorProto.FLOAT, [1], [float(target_size)])
    )
    node_list.extend(
        [
            helper.make_node("Div", ["size_const", "h_float"], ["scale_h"]),
            helper.make_node("Div", ["size_const", "w_float"], ["scale_w"]),
            helper.make_node("Min", ["scale_h", "scale_w"], ["scale"]),
            helper.make_node("Mul", ["h_float", "scale"], ["new_hf"]),
            helper.make_node("Mul", ["w_float", "scale"], ["new_wf"]),
        ]
    )

    graph.initializer.append(
        helper.make_tensor("half_const", TensorProto.FLOAT, [1], [0.5])
    )
    node_list.extend(
        [
            helper.make_node("Add", ["new_hf", "half_const"], ["new_hf_add"]),
            helper.make_node("Add", ["new_wf", "half_const"], ["new_wf_add"]),
            helper.make_node("Floor", ["new_hf_add"], ["new_h"]),
            helper.make_node("Floor", ["new_wf_add"], ["new_w"]),
            helper.make_node("Cast", ["new_h"], ["new_h_i"], to=TensorProto.INT64),
            helper.make_node("Cast", ["new_w"], ["new_w_i"], to=TensorProto.INT64),
        ]
    )

    graph.initializer.extend(
        [
            helper.make_tensor("const_c3", TensorProto.INT64, [1], [3]),
            helper.make_tensor("roi", TensorProto.FLOAT, [0], []),
            helper.make_tensor("resize_scales", TensorProto.FLOAT, [0], []),
        ]
    )
    node_list.append(
        helper.make_node(
            "Concat",
            inputs=["new_h_i", "new_w_i", "const_c3"],
            outputs=["resize_sizes"],
            axis=0,
        )
    )
    node_list.append(
        helper.make_node(
            "Resize",
            inputs=[inter, "roi", "resize_scales", "resize_sizes"],
            outputs=["resize_out"],
            mode="linear",
            coordinate_transformation_mode="pytorch_half_pixel",
        )
    )
    inter = "resize_out"

    # 4. Padding to target size with black border
    graph.initializer.extend(
        [
            helper.make_tensor("const_640", TensorProto.INT64, [1], [target_size]),
            helper.make_tensor("pad_zero", TensorProto.INT64, [1], [0]),
            helper.make_tensor("pad_val", TensorProto.FLOAT, [], [0.0]),
        ]
    )
    # 单边填充（仅底部和右侧），pad_top=0, pad_left=0
    node_list.extend(
        [
            helper.make_node("Sub", ["const_640", "new_h_i"], ["pad_h"]),  # pad_bottom
            helper.make_node("Sub", ["const_640", "new_w_i"], ["pad_w"]),  # pad_right
            helper.make_node(
                "Concat",
                inputs=["pad_zero", "pad_zero", "pad_zero", "pad_h", "pad_w", "pad_zero"],
                outputs=["pads"],
                axis=0,
            ),
            helper.make_node("Pad", [inter, "pads", "pad_val"], ["pad_out"], mode="constant"),
        ]
    )
    inter = "pad_out"

    # 5. Normalize
    graph.initializer.append(
        helper.make_tensor("scale_tensor", TensorProto.FLOAT, [], [255.0])
    )
    node_list.append(helper.make_node("Div", [inter, "scale_tensor"], ["normed"]))
    inter = "normed"

    # graph.initializer.extend(
    #     [
    #         helper.make_tensor("mean_tensor", TensorProto.FLOAT, [3], list(mean)),
    #         helper.make_tensor("std_tensor", TensorProto.FLOAT, [3], list(std)),
    #         helper.make_tensor("mean_shape", TensorProto.INT64, [3], [1, 1, 3]),
    #         helper.make_tensor("std_shape", TensorProto.INT64, [3], [1, 1, 3]),
    #     ]
    # )
    # node_list.extend(
    #     [
    #         helper.make_node("Reshape", ["mean_tensor", "mean_shape"], ["mean_reshaped"]),
    #         helper.make_node("Reshape", ["std_tensor", "std_shape"], ["std_reshaped"]),
    #         helper.make_node("Sub", [inter, "mean_reshaped"], ["mean_sub"]),
    #         helper.make_node("Div", ["mean_sub", "std_reshaped"], ["stded"]),
    #     ]
    # )
    # inter = "stded"

    # 6. HWC -> CHW
    node_list.append(
        helper.make_node(
            "Transpose",
            inputs=[inter],
            outputs=["transpose_out"],
            perm=[2, 0, 1],
        )
    )
    inter = "transpose_out"

    # 7. Add batch dimension
    node_list.append(
        helper.make_node(
            "Unsqueeze",
            inputs=[inter],
            outputs=[orig_input_name],
            axes=[0],
        )
    )

    original_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(node_list + original_nodes)

    return model


def convert(
    input_path: str,
    output_path: str,
    target_size: int = 640,
    mean=(0.502, 0.502, 0.502),
    std=(0.5, 0.5, 0.5),
    bgr2rgb: bool = True,
) -> None:
    model = onnx.load(input_path)
    model = insert_preprocess_nodes(
        model,
        target_size=target_size,
        mean=mean,
        std=std,
        bgr2rgb=bgr2rgb,
    )
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"✅ 新模型已保存: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Insert image preprocessing into an ONNX model.")
    parser.add_argument("--input", help="原始 ONNX 模型路径")
    parser.add_argument("--output", help="输出 ONNX 模型路径")
    parser.add_argument("--size", type=int, default=640, help="目标尺寸，默认 640")
    parser.add_argument(
        "--no-bgr2rgb",
        dest="bgr2rgb",
        action="store_false",
        help="禁用 BGR->RGB 转换",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output, target_size=args.size, bgr2rgb=args.bgr2rgb)
    
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
