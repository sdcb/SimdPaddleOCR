using System.Text.Json.Nodes;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.ModelProvider;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Medium;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Small;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;

namespace Sdcb.SimdPaddleOCR.Tests;

interface IBenchEngine : IDisposable
{
    string Name { get; }
    JsonObject Extra { get; }
    string LoadedMessage(double workingSetMb);
    BenchEngineOutput Run(byte[] bgr, int width, int height, int stride);
}

sealed class BenchEngineOutput
{
    public int Detected { get; init; }
    public string[] Texts { get; init; } = [];
    public int[] Rotations { get; init; } = [];
    /// <summary>Per-line AABB [minX, minY, maxX, maxY] aligned with Texts/Rotations.</summary>
    public float[][] Boxes { get; init; } = [];
    public string? Hash { get; init; }
    public Dictionary<string, double>? StageMs { get; init; }
    public Dictionary<string, long>? StageCalls { get; init; }
    public Dictionary<string, BenchmarkMetric>? OperatorMs { get; init; }
    public Dictionary<string, BenchmarkMetric>? ConvClassMs { get; init; }
}

static class BenchBoxes
{
    public static float[] Aabb(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4)
        =>
        [
            Math.Min(Math.Min(x1, x2), Math.Min(x3, x4)),
            Math.Min(Math.Min(y1, y2), Math.Min(y3, y4)),
            Math.Max(Math.Max(x1, x2), Math.Max(x3, x4)),
            Math.Max(Math.Max(y1, y2), Math.Max(y3, y4)),
        ];
}

static class BenchEngines
{
    public static IBenchEngine Create(string engine, string modelType, int workers, string cAssetsDir, BenchEngineSettings settings) => engine switch
    {
        // sharp pins Cpu so historical baselines stay comparable; "auto" lets
        // the factory pick (Vulkan when a usable device exists).
        "sharp" => new SharpEngine(modelType, workers, OcrBackend.Cpu, settings.DetLimitSideLength),
        "vulkan" => new SharpEngine(modelType, workers, OcrBackend.Vulkan, settings.DetLimitSideLength),
        "metal" => new SharpEngine(modelType, workers, OcrBackend.Metal, settings.DetLimitSideLength),
        "auto" => new SharpEngine(modelType, workers, OcrBackend.Auto, settings.DetLimitSideLength),
        "c" => new CEngine(cAssetsDir, workers, modelType),
#if RAPIDOCR
        "rapid" => new RapidEngine(modelType, workers, cuda: true, settings.RapidModelsDir, settings.RapidDetLimitSideLength, settings.RapidImgResize),
        "rapid-cpu" => new RapidEngine(modelType, workers, cuda: false, settings.RapidModelsDir, settings.RapidDetLimitSideLength, settings.RapidImgResize),
#else
        "rapid" or "rapid-cpu" => throw new InvalidOperationException(
            $"--engine {engine} needs a build with RapidOcrNet: add -p:EnableRapidOcr=true (pulls the ONNX Runtime CUDA package)."),
#endif
        _ => throw new ArgumentException("--engine must be sharp, vulkan, metal, auto, c, rapid, or rapid-cpu"),
    };

    public static PaddleOcrModelBundle Bundle(string modelType) => modelType switch
    {
        "tiny" => ChineseV6TinyModels.Default,
        "small" => ChineseV6SmallModels.Default,
        "medium" => ChineseV6MediumModels.Default,
        _ => throw new ArgumentException("--model must be tiny, small, or medium"),
    };
}

sealed record BenchEngineSettings
{
    /// <summary>Short side the hand-written detector resizes to (--det-limit).</summary>
    public int DetLimitSideLength { get; init; } = 960;

    /// <summary>Directory holding the PP-OCRv6 det/cls/rec weights RapidOcrNet loads (--rapid-models).</summary>
    public string RapidModelsDir { get; init; } = "models";

    /// <summary>Short side RapidOcrNet's detector resizes to (--rapid-det-limit).</summary>
    public int RapidDetLimitSideLength { get; init; } = 736;

    /// <summary>
    /// RapidOcrNet's legacy max-side cap (--rapid-img-resize). When set it replaces the
    /// short-side rule, which is how the detector input is matched to this library's
    /// <see cref="DetLimitSideLength"/>.
    /// </summary>
    public int RapidImgResize { get; init; }
}
