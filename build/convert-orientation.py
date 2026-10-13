"""下载固定版本的 Paddle 方向模型，导出 ONNX，独立简化并验证数值后写入交付模型。

在仓库根目录使用隔离的 Python 3.12 环境运行：
  pip install paddlepaddle==3.1.1 paddle2onnx==2.1.0 onnx==1.17.0 onnxsim==0.8.1
  pip install opencv-python-headless==4.13.0.92 opt-einsum==3.3.0 polygraphy==0.49.26
  python build/convert-orientation.py --endpoint https://hf-mirror.com
默认下载地址为 Hugging Face；--endpoint 可指定镜像，--source-dir 指定原始模型目录，
--sample 指定四方向测试图片。现有 OCR 模型不替换，数值校验失败时不写入该模型的交付 ONNX。
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import requests

ROOT = Path(__file__).resolve().parent.parent
# 固定来源提交、模型输入高宽和交付文件名；纸张模型四分类，文本行模型仅两分类。
SOURCES = {
    "PP-LCNet_x1_0_doc_ori": ("d3b95a6dff5fe8a94f2748e12b61cb26818a0df8", (224, 224), "doc_ori.onnx"),
    "PP-LCNet_x1_0_textline_ori": ("cd237a44b0e359d4fe38310a416203cf7403faa5", (80, 160), "textline_ori_x1.onnx"),
}


def digest(path):
    """计算源文件或交付文件的 SHA-256，供转换记录追溯来源。

    参数：path 为待读取的 pathlib.Path。
    返回：小写十六进制哈希字符串。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preprocess(bgr, document):
    """按官方分类预处理生成连续 float32 NCHW 输入。

    参数：bgr 为 H×W×3 的八位 BGR 图像；document 为 True 时执行纸张模型的
    短边 256 缩放及中心 224 裁剪，为 False 时按文本行模型缩放至宽 160、高 80。
    返回：RGB 归一化后的 [1, 3, H, W] NumPy 数组。
    """
    rgb = bgr[:, :, ::-1]
    if document:
        h, w = rgb.shape[:2]
        scale = 256 / min(h, w)
        rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
        rgb = rgb[(h - 224) // 2:(h - 224) // 2 + 224, (w - 224) // 2:(w - 224) // 2 + 224]
    else:
        rgb = cv2.resize(rgb, (160, 80), interpolation=cv2.INTER_LINEAR)
    data = rgb.astype(np.float32)
    # 按 PaddleX 的顺序执行 float32 乘法再加法，避免把舍入差异误认为转换误差。
    for c, (mean, std) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
        data[:, :, c] *= np.float32(1 / 255 / std)
        data[:, :, c] += np.float32(-mean / std)
    return np.ascontiguousarray(data.transpose(2, 0, 1)[None])


def session(path):
    """创建不启用运行时图优化的 ONNX Runtime CPU 会话。

    参数：path 为原始或简化后的 ONNX 文件路径。
    返回：使用 4 个算子线程的 CPU 会话，直接验证文件中的图，不让运行时代为修复。
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    # 比较文件中的原图和简化图，禁用 ORT 图优化，避免运行时改变待验证节点。
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def main():
    """解析命令行，下载并验证两个方向模型，最后保存模型和转换记录。

    命令行参数：--endpoint 为下载服务根地址；--source-dir 为原始 Paddle 文件目录；
    --sample 为用于 0/90/180/270 度比对的样例图片。无函数参数和返回值。
    任意文件、形状或数值检查失败时抛出异常，不写入尚未验证通过的模型。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--source-dir", type=Path, default=ROOT / ".tmp-artifacts/orientation-source")
    parser.add_argument("--sample", type=Path, default=ROOT / "examples/sample.jpg")
    args = parser.parse_args()
    scratch = ROOT / ".tmp-artifacts/orientation-validation"
    scratch.mkdir(parents=True, exist_ok=True)
    # Paddle 导入数据集模块时尝试写入用户目录；临时触发其只读目录回退，改用工作区临时目录。
    tempfile.tempdir = str(scratch)
    original_access = os.access
    profile = os.path.expanduser("~")
    with patch("os.access", lambda p, m: False if str(p) == profile else original_access(p, m)):
        import paddle
        import paddle2onnx
    from onnxsim import simplify

    image = cv2.imread(str(args.sample))
    if image is None:
        raise ValueError(f"Cannot read {args.sample}")
    report = {"tools": {p: importlib.metadata.version(p) for p in
                       ["paddlepaddle", "paddle2onnx", "onnx", "onnxsim", "onnxruntime", "opencv-python-headless"]},
              "sample": {"path": str(args.sample.relative_to(ROOT)), "sha256": digest(args.sample)}, "models": {}}
    fixtures = []
    rng = np.random.default_rng(137)
    for name, (revision, (h, w), destination) in SOURCES.items():
        source = args.source_dir / name
        source.mkdir(parents=True, exist_ok=True)
        source_info = {}
        for filename in ["inference.json", "inference.pdiparams", "inference.yml", "config.json", "README.md"]:
            path = source / filename
            if not path.exists():
                response = requests.get(f"{args.endpoint.rstrip('/')}/PaddlePaddle/{name}/resolve/{revision}/{filename}", timeout=90)
                response.raise_for_status()
                path.write_bytes(response.content)
            source_info[filename] = {"sha256": digest(path), "bytes": path.stat().st_size}
        raw = scratch / (name + ".raw.onnx")
        simplified = scratch / (name + ".sim.onnx")
        # 导出器不启用内置优化；图结构和参数必须成对使用，随后再交给独立 onnxsim 简化。
        paddle2onnx.export(str(source / "inference.json"), str(source / "inference.pdiparams"),
                          save_file=str(raw), opset_version=13, optimize_tool=None)
        raw_model = onnx.load(raw)
        onnx.checker.check_model(raw_model, full_check=True)
        input_name = raw_model.graph.input[0].name
        # 固定真实输入形状，执行独立简化、严格形状推导和完整 ONNX 校验。
        sim, checked = simplify(raw_model, overwrite_input_shapes={input_name: [1, 3, h, w]})
        if not checked:
            raise AssertionError("onnxsim rejected its simplified graph")
        sim = onnx.shape_inference.infer_shapes(sim, strict_mode=True)
        onnx.checker.check_model(sim, full_check=True)
        onnx.save(sim, simplified)
        print(name, "nodes", len(raw_model.graph.node), "->", len(sim.graph.node), Counter(n.op_type for n in sim.graph.node), flush=True)
        config = paddle.inference.Config(str(source / "inference.json"), str(source / "inference.pdiparams"))
        config.disable_gpu()
        config.disable_mkldnn()
        config.set_cpu_math_library_num_threads(4)
        predictor = paddle.inference.create_predictor(config)
        raw_session, sim_session = session(raw), session(simplified)
        # 随机输入、常量输入和样例四方向共同覆盖数值和标签；固定随机种子确保可重复。
        cases = [(f"random-{i}", rng.uniform(-2.2, 2.7, (1, 3, h, w)).astype(np.float32)) for i in range(8)]
        cases += [(f"constant-{v}", np.full((1, 3, h, w), v, np.float32)) for v in [-2.0, 0.0, 2.0]]
        cases += [(f"sample-{angle}", preprocess(np.ascontiguousarray(np.rot90(image, -angle // 90)), h == 224)) for angle in [0, 90, 180, 270]]
        comparisons = []
        for case_name, data in cases:
            handle = predictor.get_input_handle(predictor.get_input_names()[0])
            handle.reshape(data.shape)
            handle.copy_from_cpu(data)
            predictor.run()
            expected = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
            exported = raw_session.run(None, {input_name: data})[0]
            optimized = sim_session.run(None, {input_name: data})[0]
            # 原始 Paddle 是独立参考；原 ONNX、简化 ONNX 都必须满足误差上限且最高分类一致。
            for actual in [exported, optimized]:
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
                assert np.argmax(actual) == np.argmax(expected), (case_name, actual, expected)
            np.testing.assert_allclose(optimized, exported, rtol=1e-5, atol=2e-6)
            comparisons.append({"case": case_name, "paddle_raw_max_abs": float(np.max(np.abs(expected - exported))),
                                "paddle_sim_max_abs": float(np.max(np.abs(expected - optimized))),
                                "raw_sim_max_abs": float(np.max(np.abs(exported - optimized))),
                                "label": int(np.argmax(expected)), "scores": expected.tolist()})
            if h == 224:
                tensor_path = scratch / (case_name + ".f32")
                data.tofile(tensor_path)
                fixtures.append({"case": case_name, "input": tensor_path.name, "scores": expected.reshape(-1).tolist()})
        # 只有该模型的全部独立比较通过，才将简化后的文件写入交付目录。
        output = ROOT / "models" / destination
        output.write_bytes(simplified.read_bytes())
        report["models"][name] = {"repository": f"https://huggingface.co/PaddlePaddle/{name}", "revision": revision,
            "source_files": source_info, "output": destination, "sha256": digest(output),
            "raw_nodes": len(raw_model.graph.node), "simplified_nodes": len(sim.graph.node),
            "operators": dict(Counter(n.op_type for n in sim.graph.node)), "comparisons": comparisons}
    (scratch / "fixtures.json").write_text(json.dumps(fixtures, indent=2), encoding="utf-8")
    (ROOT / "models/orientation-manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Paddle -> raw ONNX -> onnxsim numerical checks passed; model assets written.", flush=True)


if __name__ == "__main__":
    main()
