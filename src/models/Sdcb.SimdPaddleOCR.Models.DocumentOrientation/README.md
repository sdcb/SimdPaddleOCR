# Sdcb.SimdPaddleOCR.Models.DocumentOrientation

Independent PP-LCNet_x1_0_doc_ori model for whole-page orientation (0°, 90°, 180°, 270°). This package embeds the simplified ONNX model and its conversion/validation manifest. It depends only on `Sdcb.SimdPaddleOCR.ModelProvider`; reference the inference core and your OCR model bundle separately.

```csharp
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.Models.DocumentOrientation;

using var ocr = await PaddleOcrAll.LoadAsync(
    ChineseV6TinyModels.Default,
    DocumentOrientationModel.Provider,
    options: null);
var result = ocr.Run(pixels, width, height);
Console.WriteLine(result.DocumentOrientation?.OrientationDegrees);
```

The optional stage runs before text detection. It corrects the page by the predicted angle and returns OCR boxes in the original input coordinates. The core does not depend on this model package. To read the model alone, use `DocumentOrientationModel.OpenRead()` or `DocumentOrientationModel.Provider.OpenReadAsync()`.

The source Paddle inference.json + pdiparams model is exported with Paddle2ONNX, independently simplified using onnxsim (310 → 113 nodes), and numerically compared against original Paddle inference. It uses RGB, short-side resize 256, center crop 224, and ImageNet normalization. The manifest records source revisions, hashes, tool versions and per-case errors.

Source: [PaddlePaddle/PP-LCNet_x1_0_doc_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori), Apache-2.0. Copyright (c) PaddlePaddle Authors. [Integration, validation and benchmarks](https://github.com/sdcb/SimdPaddleOCR/blob/master/docs/document-orientation.md).
