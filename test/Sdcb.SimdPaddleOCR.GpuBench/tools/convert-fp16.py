"""fp32 -> fp16 ONNX converter for the ORT fp16 experiment.

Keeps the graph's inputs/outputs in fp32 (keep_io_types=True) so a caller that feeds
fp32 tensors — RapidOcrNet does — can load the model unchanged; the graph inserts its
own casts.

    pip install onnx onnxconverter-common
    python convert-fp16.py <src.onnx> <dst.onnx>

Caveats found on this repo's PP-OCRv6 models (see docs/rapidocr-cuda.md):
  * rec / cls convert cleanly and keep per-line accuracy identical,
  * the DB detector does not: the converted tiny det returns no boxes at all,
  * on ONNX Runtime CUDA the fp16 rec is *slower* than fp32 once input shapes churn.
"""
import sys

import onnx
from onnxconverter_common import float16


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    model = onnx.load(src)
    converted = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=False)
    onnx.checker.check_model(converted)
    onnx.save(converted, dst)
    halves = sum(1 for i in converted.graph.initializer if i.data_type == onnx.TensorProto.FLOAT16)
    print(f"{src} -> {dst}: fp16 initializers={halves}, size={len(converted.SerializeToString()) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
