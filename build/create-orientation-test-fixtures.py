"""生成固定、体积较小的 Gemm ONNX 用例和 OpenCV 预处理黄金值。

从仓库根目录运行，无命令行参数。输出到单元测试 Assets/Orientation 目录，
用于独立验证矩阵转置、缩放、偏置广播及纸张模型的 RGB 中心裁剪归一化。
"""
import json
from pathlib import Path
import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper

root = Path(__file__).resolve().parent.parent
out = root / "test/Sdcb.SimdPaddleOCR.UnitTests/Assets/Orientation"
out.mkdir(parents=True, exist_ok=True)
cases = []
# 分别覆盖普通偏置、双输入转置及缩放、列偏置广播，以及完全不提供偏置。
for name, trans_a, trans_b, alpha, beta, bias_shape in [
    ("gemm", 0, 0, 1.0, 1.0, [2]),
    ("gemm-transpose-scale", 1, 1, 0.75, 1.5, [2]),
    ("gemm-column-bias", 0, 1, -0.5, 0.25, [2, 1]),
    ("gemm-no-bias", 0, 0, 2.0, 1.0, None),
]:
    a = np.arange(6, dtype=np.float32).reshape(2, 3) - 2
    b = (np.arange(6, dtype=np.float32).reshape(3, 2) + 1) / 4
    raw_a = a.T.copy() if trans_a else a
    raw_b = b.T.copy() if trans_b else b
    bias = np.arange(2, dtype=np.float32).reshape(bias_shape) + 0.25 if bias_shape else None
    nodes = []
    inputs = ["x", "w"]
    if bias is not None:
        # 故意让 TensorProto 的描述名称与 Constant 输出边不同，复现简化器可能生成的命名。
        nodes.append(helper.make_node("Constant", [], ["bias-edge"], value=numpy_helper.from_array(bias, name="descriptive-bias")))
        inputs.append("bias-edge")
    nodes.append(helper.make_node("Gemm", inputs, ["y"], transA=trans_a, transB=trans_b, alpha=alpha, beta=beta))
    graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info("x", 1, raw_a.shape)],
        [helper.make_tensor_value_info("y", 1, [2, 2])], [numpy_helper.from_array(raw_b, name="w")])
    model = onnx.shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8))
    onnx.checker.check_model(model)
    onnx.save(model, out / (name + ".onnx"))
    expected = alpha * (a @ b) + (beta * bias if bias is not None else 0)
    cases.append({"name": name, "shape": list(raw_a.shape), "input": raw_a.ravel().tolist(), "expected": expected.ravel().tolist()})
(out / "gemm.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
# 非正方形的确定性输入能覆盖缩放后的中心裁剪偏移，黄金值由 OpenCV 独立生成。
width, height = 17, 19
bgr = ((np.arange(width * height * 3, dtype=np.int32) * 37 + 11) % 256).astype(np.uint8).reshape(height, width, 3)
rgb = cv2.resize(bgr[:, :, ::-1], (round(width * 256 / min(width, height)), round(height * 256 / min(width, height))))
y, x = (rgb.shape[0] - 224) // 2, (rgb.shape[1] - 224) // 2
rgb = rgb[y:y + 224, x:x + 224].astype(np.float32)
for c, (mean, std) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
    rgb[:, :, c] *= np.float32(1 / 255 / std)
    rgb[:, :, c] += np.float32(-mean / std)
# 采样裁剪边界和内部位置，测试时按 NCHW/NHWC 分别索引同一套 RGB 黄金值。
samples = [{"channel": c, "x": x, "y": y, "value": float(rgb[y, x, c])}
    for c in range(3) for x in [0, 1, 16, 57, 112, 173, 222, 223] for y in [0, 7, 111, 222, 223]]
(out / "preprocess.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
