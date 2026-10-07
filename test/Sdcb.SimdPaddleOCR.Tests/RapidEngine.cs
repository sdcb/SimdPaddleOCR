#if RAPIDOCR
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Microsoft.ML.OnnxRuntime;
using RapidOcrNet;
using SkiaSharp;

namespace Sdcb.SimdPaddleOCR.Tests;

/// <summary>
/// RapidOcrNet (ONNX Runtime, optional CUDA) behind the same harness interface
/// as <see cref="SharpEngine"/>. It loads the repository's own PP-OCRv6 weights,
/// so the comparison isolates the engine instead of the model.
/// </summary>
sealed class RapidEngine : IBenchEngine
{
    private readonly RapidOcr _ocr = new();
    private readonly RapidOcrOptions _options;
    private readonly string _provider;
    private SKBitmap? _bitmap;
    private byte[] _bgra = [];

    public RapidEngine(string modelType, int workers, bool cuda, string modelsDir, int detLimitSideLength, int imgResize = 0)
    {
        (string det, string rec, string keys) = modelType switch
        {
            "tiny" => ("det.onnx", "rec.onnx", "ppocr_keys.txt"),
            "small" => ("small_det.onnx", "small_rec.onnx", "rec_keys.txt"),
            "medium" => ("medium_det.onnx", "medium_rec.onnx", "rec_keys.txt"),
            _ => throw new ArgumentException("--model must be tiny, small, or medium"),
        };
        string detPath = Path.Combine(modelsDir, det);
        string clsPath = Path.Combine(modelsDir, "cls.onnx");
        string recPath = Path.Combine(modelsDir, rec);
        string keysPath = Path.Combine(modelsDir, keys);
        foreach (string path in new[] { detPath, clsPath, recPath, keysPath })
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"--rapid-models is missing {path} (see the repository's models/ directory).", path);
        }

        // Presets only supply the detector normalization; every path is overridden.
        RapidOcrModelSet models = (modelType switch
        {
            "tiny" => RapidOcrModelSet.PPOCRv6Tiny,
            "small" => RapidOcrModelSet.PPOCRv6Small,
            _ => RapidOcrModelSet.PPOCRv6Medium,
        }) with
        {
            DetModelPath = detPath,
            ClsModelPath = clsPath,
            RecModelPath = recPath,
            KeysPath = keysPath,
        };

        SessionOptions sessionOptions = RapidOcr.GetDefaultSessionOptions();
        _provider = cuda ? "cuda" : "cpu";
        if (cuda)
        {
            try
            {
                sessionOptions.AppendExecutionProvider_CUDA();
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    "CUDA execution provider is unavailable; use --engine rapid-cpu or fix the GPU runtime.", ex);
            }
        }
        _ocr.InitModels(models, sessionOptions);

        _options = RapidOcrOptions.PPOCRv6 with
        {
            LimitSideLen = detLimitSideLength,
            ImgResize = imgResize,
            RecMaxDegreeOfParallelism = Math.Max(1, workers),
        };

        Extra["provider"] = _provider;
        Extra["rapidVersion"] = typeof(RapidOcr).Assembly.GetName().Version?.ToString();
        Extra["rapidDetLimitSideLen"] = detLimitSideLength;
        Extra["rapidImgResize"] = imgResize;
        Extra["rapidRecParallelism"] = Math.Max(1, workers);
        Extra["rapidDet"] = detPath;
        Extra["rapidCls"] = clsPath;
        Extra["rapidRec"] = recPath;
        Extra["rapidKeys"] = keysPath;
        Extra["rapidOptions"] = nameof(RapidOcrOptions.PPOCRv6);
    }

    public string Name => "rapid";
    public JsonObject Extra { get; } = [];
    public string LoadedMessage(double workingSetMb) =>
        $"loaded working_set={workingSetMb:F1} MB engine=rapid provider={_provider} " +
        $"rec_parallelism={_options.RecMaxDegreeOfParallelism} det_limit={_options.LimitSideLen} " +
        $"img_resize={_options.ImgResize} " +
        $"rapid={typeof(RapidOcr).Assembly.GetName().Version} ort={typeof(SessionOptions).Assembly.GetName().Version}";

    public BenchEngineOutput Run(byte[] bgr, int width, int height, int stride)
    {
        // RapidOcrNet only accepts an SKBitmap; the harness hands every engine the
        // same BGR24 buffer, so the copy is measured as its own stage.
        long converted = Stopwatch.GetTimestamp();
        SKBitmap bitmap = Bitmap(width, height);
        Convert(bgr, width, height, stride);
        Marshal.Copy(_bgra, 0, bitmap.GetPixels(), _bgra.Length);
        long detected = Stopwatch.GetTimestamp();

        OcrResult result = _ocr.Detect(bitmap, _options);
        long finished = Stopwatch.GetTimestamp();

        double clsMs = 0, recMs = 0;
        foreach (TextBlock block in result.TextBlocks)
        {
            clsMs += block.AngleTime;
            recMs += block.CrnnTime;
        }
        // RapidOcrNet's own clocks: DbNetTime is the detector (tensor fill, run and
        // contours), DetectTime is the whole DetectOnce call, i.e. det + cls + rec.
        // AngleTime/CrnnTime are per crop and include that crop's resize.
        Dictionary<string, double> stages = new()
        {
            ["input_convert"] = Ms(converted, detected),
            ["det_graph"] = result.DbNetTime,
            ["cls_graph"] = clsMs,
            ["rec_graph"] = recMs,
            ["pipeline_wall"] = result.DetectTime,
            ["detect_call"] = Ms(detected, finished),
        };
        return new BenchEngineOutput
        {
            Detected = result.TextBlocks.Length,
            Texts = result.TextBlocks.Select(x => x.Text).ToArray(),
            Rotations = result.TextBlocks.Select(x => x.AngleIndex == 1 ? 180 : 0).ToArray(),
            Boxes = result.TextBlocks.Select(x => BenchBoxes.Aabb(
                x.BoxPoints[0].X, x.BoxPoints[0].Y,
                x.BoxPoints[1].X, x.BoxPoints[1].Y,
                x.BoxPoints[2].X, x.BoxPoints[2].Y,
                x.BoxPoints[3].X, x.BoxPoints[3].Y)).ToArray(),
            StageMs = stages,
        };
    }

    private SKBitmap Bitmap(int width, int height)
    {
        if (_bitmap is { } cached && cached.Width == width && cached.Height == height)
            return cached;
        _bitmap?.Dispose();
        _bitmap = new SKBitmap(new SKImageInfo(width, height, SKColorType.Bgra8888, SKAlphaType.Opaque));
        _bgra = new byte[width * height * 4];
        return _bitmap;
    }

    private unsafe void Convert(byte[] bgr, int width, int height, int stride)
    {
        fixed (byte* source = bgr)
        {
            fixed (byte* target = _bgra)
            {
                for (int y = 0; y < height; y++)
                {
                    byte* row = source + y * stride;
                    byte* destination = target + y * width * 4;
                    for (int x = 0; x < width; x++)
                    {
                        destination[0] = row[0];
                        destination[1] = row[1];
                        destination[2] = row[2];
                        destination[3] = 255;
                        row += 3;
                        destination += 4;
                    }
                }
            }
        }
    }

    private static double Ms(long from, long to) => (to - from) * 1000.0 / Stopwatch.Frequency;

    public void Dispose()
    {
        _ocr.Dispose();
        _bitmap?.Dispose();
    }
}
#endif
