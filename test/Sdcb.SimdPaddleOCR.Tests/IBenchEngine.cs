using System.Text.Json.Nodes;
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
    public string? Hash { get; init; }
    public Dictionary<string, double>? StageMs { get; init; }
    public Dictionary<string, long>? StageCalls { get; init; }
    public Dictionary<string, BenchmarkMetric>? OperatorMs { get; init; }
    public Dictionary<string, BenchmarkMetric>? ConvClassMs { get; init; }
}

static class BenchEngines
{
    public static IBenchEngine Create(string engine, string modelType, int workers, string cAssetsDir) => engine switch
    {
        "sharp" => new SharpEngine(modelType, workers),
        "c" => new CEngine(cAssetsDir, workers, modelType),
        "openvino" => new OpenVinoEngine(modelType),
        _ => throw new ArgumentException("--engine must be sharp, c, or openvino"),
    };

    public static PaddleOcrModelBundle Bundle(string modelType) => modelType switch
    {
        "tiny" => ChineseV6TinyModels.Default,
        "small" => ChineseV6SmallModels.Default,
        "medium" => ChineseV6MediumModels.Default,
        _ => throw new ArgumentException("--model must be tiny, small, or medium"),
    };
}
