"""Fixtures for test/Sdcb.SimdPaddleOCR.UnitTests/DocumentOrientationTests.cs.

Run from anywhere (needs onnx, numpy, opencv-python); no arguments.
- Assets/Orientation/preprocess.json: OpenCV golden values of the doc_ori preprocessing
  (short side 256, center crop 224, RGB ImageNet normalization) on a 17 x 19 pattern.
- Fixtures/DocOrientationTiny4.onnx.gz / DocOrientationTiny2.onnx.gz: tiny stand-ins for the
  real 6.7 MB model, [1,3,224,224] -> GlobalAveragePool -> Reshape -> Gemm -> Softmax (the real
  model's head). With the four-class weights a solid red / green / blue / black page scores class 0 / 1 / 2 / 3.

Based on the fixture script of sdcb/SimdPaddleOCR#32 by hnlyf168.
"""
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper

root = Path(__file__).resolve().parent.parent / "test/Sdcb.SimdPaddleOCR.UnitTests"
assets = root / "Assets/Orientation"
fixtures = root / "Fixtures"
assets.mkdir(parents=True, exist_ok=True)
fixtures.mkdir(parents=True, exist_ok=True)

# non-square deterministic input so the crop offset after the resize is exercised
width, height = 17, 19
bgr = ((np.arange(width * height * 3, dtype=np.int32) * 37 + 11) % 256).astype(np.uint8).reshape(height, width, 3)
rgb = cv2.resize(bgr[:, :, ::-1], (round(width * 256 / min(width, height)), round(height * 256 / min(width, height))))
y, x = (rgb.shape[0] - 224) // 2, (rgb.shape[1] - 224) // 2
rgb = rgb[y:y + 224, x:x + 224].astype(np.float32)
for c, (mean, std) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
    rgb[:, :, c] *= np.float32(1 / 255 / std)
    rgb[:, :, c] += np.float32(-mean / std)
samples = [{"channel": c, "x": x, "y": y, "value": float(rgb[y, x, c])}
           for c in range(3) for x in [0, 1, 16, 57, 112, 173, 222, 223] for y in [0, 7, 111, 222, 223]]
(assets / "preprocess.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")


def tiny(classes: int) -> bytes:
    # channel means (R, G, B) of the normalized input -> logits; the last class rises when all are low
    w = np.zeros((3, classes), dtype=np.float32)
    for c in range(min(3, classes)):
        w[c, c] = 10
    if classes == 4:
        w[:, 3] = -10
    nodes = [
        helper.make_node("GlobalAveragePool", ["x"], ["pool"]),
        helper.make_node("Reshape", ["pool", "shape"], ["flat"]),
        helper.make_node("Gemm", ["flat", "w", "b"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["y"], axis=1),
    ]
    graph = helper.make_graph(
        nodes, f"doc-ori-tiny-{classes}",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, classes])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(np.zeros(classes, np.float32), "b"),
         numpy_helper.from_array(np.array([1, 3], np.int64), "shape")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    return model.SerializeToString()


for classes in (4, 2):
    (fixtures / f"DocOrientationTiny{classes}.onnx.gz").write_bytes(gzip.compress(tiny(classes), mtime=0))
