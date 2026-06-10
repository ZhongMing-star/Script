import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

def merge_preprocess_with_letterbox(
    origin_onnx_path: str,
    output_onnx_path: str,
    target_size: int = 640,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
    bgr2rgb: bool = True,
    scale: float = 255.0
):
    model = onnx.load(origin_onnx_path)
    graph = model.graph
    orig_input_name = graph.input[0].name
    node_list = []
    inter = "raw_image"

    # 1. 动态输入: [H, W, 3] uint8
    new_input_name = "raw_image"
    new_input = helper.make_tensor_value_info(
        new_input_name,
        TensorProto.UINT8,
        ["H", "W", 3]
    )
    graph.input[0].CopyFrom(new_input)

    # ==================== 统一常量(全部一维/标量，适配Slice/Concat) ====================
    slice_0 = helper.make_tensor("slice_0", TensorProto.INT64, [1], [0])
    slice_1 = helper.make_tensor("slice_1", TensorProto.INT64, [1], [1])
    slice_2 = helper.make_tensor("slice_2", TensorProto.INT64, [1], [2])

    const_2f = helper.make_tensor("const_2f", TensorProto.FLOAT, [], [2.0])
    pad_zero_1d = helper.make_tensor("pad_zero_1d", TensorProto.INT64, [1], [0])
    one_1d = helper.make_tensor("one_1d", TensorProto.INT64, [1], [1])

    target_tensor = helper.make_tensor("target_tensor", TensorProto.FLOAT, [], [target_size])
    pad_val = helper.make_tensor("pad_val", TensorProto.UINT8, [], [0])
    scale_tensor = helper.make_tensor("scale_tensor", TensorProto.FLOAT, [], [scale])

    # Resize 占位空张量(opset11 必须4输入)
    empty_roi = helper.make_tensor("empty_roi", TensorProto.FLOAT, [0], [])
    empty_scales = helper.make_tensor("empty_scales", TensorProto.FLOAT, [0], [])

    graph.initializer.extend([
        slice_0, slice_1, slice_2, const_2f, pad_zero_1d, one_1d,
        target_tensor, pad_val, scale_tensor, empty_roi, empty_scales
    ])

    # ==================== 2. 读取图像高宽 ====================
    shape_node = helper.make_node("Shape", [inter], ["img_shape"])
    node_list.append(shape_node)

    h_node = helper.make_node("Slice", ["img_shape", "slice_0", "slice_1"], ["img_h"])
    node_list.append(h_node)
    w_node = helper.make_node("Slice", ["img_shape", "slice_1", "slice_2"], ["img_w"])
    node_list.append(w_node)

    h_float = helper.make_node("Cast", ["img_h"], ["h_float"], to=TensorProto.FLOAT)
    w_float = helper.make_node("Cast", ["img_w"], ["w_float"], to=TensorProto.FLOAT)
    node_list.extend([h_float, w_float])

    # ==================== 3. 计算等比例缩放尺寸 ====================
    max_hw = helper.make_node("Max", ["h_float", "w_float"], ["max_hw"])
    node_list.append(max_hw)
    scale_ratio = helper.make_node("Div", ["target_tensor", "max_hw"], ["scale_ratio"])
    node_list.append(scale_ratio)

    new_h = helper.make_node("Mul", ["h_float", "scale_ratio"], ["new_h"])
    new_w = helper.make_node("Mul", ["w_float", "scale_ratio"], ["new_w"])
    node_list.extend([new_h, new_w])

    new_h_int = helper.make_node("Cast", ["new_h"], ["new_h_int"], to=TensorProto.INT64)
    new_w_int = helper.make_node("Cast", ["new_w"], ["new_w_int"], to=TensorProto.INT64)
    node_list.extend([new_h_int, new_w_int])

    # Resize 只缩放空间维度 [H,W] (2维，规避通道维挤压问题)
    resize_size = helper.make_node("Concat", ["new_h_int", "new_w_int"], ["resize_size"], axis=0)
    node_list.append(resize_size)

    # ==================== 4. HWC -> CHW 再 Resize (核心修复) ====================
    # HWC [H,W,3] -> CHW [3,H,W]
    trans_hwc2chw = helper.make_node("Transpose", [inter], ["chw_img"], perm=[2, 0, 1])
    node_list.append(trans_hwc2chw)
    inter = "chw_img"

    # opset11 Resize: 输入CHW(3维), sizes=[H,W] 只缩放后两维
    resize_node = helper.make_node(
        "Resize",
        inputs=[inter, "empty_roi", "empty_scales", "resize_size"],
        outputs=["resized_chw"],
        mode="linear",
        coordinate_transformation_mode="pytorch_half_pixel"
    )
    node_list.append(resize_node)
    inter = "resized_chw"

    # CHW -> HWC 切回
    trans_chw2hwc = helper.make_node("Transpose", [inter], ["resized_hwc"], perm=[1, 2, 0])
    node_list.append(trans_chw2hwc)
    inter = "resized_hwc"

    # ==================== 5. 计算填充量 + 居中黑边填充 ====================
    pad_h_total = helper.make_node("Sub", ["target_tensor", "new_h"], ["pad_h_total"])
    pad_w_total = helper.make_node("Sub", ["target_tensor", "new_w"], ["pad_w_total"])
    node_list.extend([pad_h_total, pad_w_total])

    pad_h_top = helper.make_node("Div", ["pad_h_total", "const_2f"], ["pad_h_top"])
    pad_h_bot = helper.make_node("Sub", ["pad_h_total", "pad_h_top"], ["pad_h_bot"])
    pad_w_left = helper.make_node("Div", ["pad_w_total", "const_2f"], ["pad_w_left"])
    pad_w_right = helper.make_node("Sub", ["pad_w_total", "pad_w_left"], ["pad_w_right"])
    node_list.extend([pad_h_top, pad_h_bot, pad_w_left, pad_w_right])

    pad_h_top_i = helper.make_node("Cast", ["pad_h_top"], ["pad_h_top_i"], to=TensorProto.INT64)
    pad_h_bot_i = helper.make_node("Cast", ["pad_h_bot"], ["pad_h_bot_i"], to=TensorProto.INT64)
    pad_w_left_i = helper.make_node("Cast", ["pad_w_left"], ["pad_w_left_i"], to=TensorProto.INT64)
    pad_w_right_i = helper.make_node("Cast", ["pad_w_right"], ["pad_w_right_i"], to=TensorProto.INT64)
    node_list.extend([pad_h_top_i, pad_h_bot_i, pad_w_left_i, pad_w_right_i])

    # 构造Pad参数 [上,下,左,右,通道0,通道0]
    pad_concat = helper.make_node(
        "Concat",
        ["pad_h_top_i", "pad_h_bot_i", "pad_w_left_i", "pad_w_right_i", "pad_zero_1d", "pad_zero_1d"],
        ["pad_axes"],
        axis=0
    )
    node_list.append(pad_concat)

    pad_node = helper.make_node(
        "Pad",
        inputs=[inter, "pad_axes", "pad_val"],
        outputs=["padded_img"],
        mode="constant"
    )
    node_list.append(pad_node)
    inter = "padded_img"

    # ==================== 6. 标准前处理: Cast / BGR2RGB(改用Transpose) / 归一化 ====================
    cast_node = helper.make_node("Cast", [inter], ["cast_float"], to=TensorProto.FLOAT)
    node_list.append(cast_node)
    inter = "cast_float"

    # 【重点】弃用Gather，用Transpose实现BGR->RGB，彻底杜绝索引越界
    if bgr2rgb:
        # HWC: [H,W,B,G,R] -> [H,W,R,G,B]
        bgr2rgb_node = helper.make_node("Transpose", [inter], ["rgb_img"], perm=[0, 1, 2][::-1])
        node_list.append(bgr2rgb_node)
        inter = "rgb_img"

    # /255
    div_scale = helper.make_node("Div", [inter, "scale_tensor"], ["div_scale_out"])
    node_list.append(div_scale)
    inter = "div_scale_out"

    # 减均值 [1,1,3]
    mean_tensor = helper.make_tensor("mean_tensor", TensorProto.FLOAT, [1, 1, 3], mean)
    graph.initializer.append(mean_tensor)
    sub_mean = helper.make_node("Sub", [inter, "mean_tensor"], ["sub_mean_out"])
    node_list.append(sub_mean)
    inter = "sub_mean_out"

    # 除标准差 [1,1,3]
    std_tensor = helper.make_tensor("std_tensor", TensorProto.FLOAT, [1, 1, 3], std)
    graph.initializer.append(std_tensor)
    div_std = helper.make_node("Div", [inter, "std_tensor"], ["preprocess_out"])
    node_list.append(div_std)
    inter = "preprocess_out"

    # HWC -> CHW
    trans_final = helper.make_node("Transpose", [inter], ["transpose_out"], perm=[2, 0, 1])
    node_list.append(trans_final)
    inter = "transpose_out"

    # 增加batch维度 [C,H,W] -> [1,C,H,W]
    unsqueeze = helper.make_node("Unsqueeze", [inter], [orig_input_name], axes=[0])
    node_list.append(unsqueeze)

    # ==================== 合并节点 & 保存 ====================
    orig_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(node_list + orig_nodes)

    onnx.checker.check_model(model)
    onnx.save(model, output_onnx_path)
    print(f"✅ 模型保存成功: {output_onnx_path}")

    # 推理自测
    try:
        sess = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
        # 测试多种尺寸
        for hh, ww in [(512, 720), (800, 400), (640, 640)]:
            test_img = np.random.randint(0, 255, (hh, ww, 3), dtype=np.uint8)
            sess.run(None, {"raw_image": test_img})
        print("✅ 多尺寸图像推理测试全部通过")
    except Exception as e:
        print(f"❌ 推理异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # ========== 请修改为你的路径与参数 ==========
    ORIGIN_ONNX = "./model/det_500m.onnx"
    OUTPUT_ONNX = "./model/det_500m_letterbox.onnx"
    TARGET_SIZE = 640
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    IS_BGR2RGB = True
    PIXEL_SCALE = 255.0

    merge_preprocess_with_letterbox(
        origin_onnx_path=ORIGIN_ONNX,
        output_onnx_path=OUTPUT_ONNX,
        target_size=TARGET_SIZE,
        mean=NORM_MEAN,
        std=NORM_STD,
        bgr2rgb=IS_BGR2RGB,
        scale=PIXEL_SCALE
    )