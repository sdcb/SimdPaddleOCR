using System.Diagnostics;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using System.Text.Json.Nodes;

namespace Sdcb.SimdPaddleOCR.Tests;

sealed class SharpEngine : IBenchEngine
{
    private const int CacheEntries = 32;
    private readonly PaddleOcrAll _ocr;
    private readonly int _workers;
    private (double Milliseconds, long Calls)[] _prevStages;
    private Dictionary<string, (double Ms, long Calls)> _prevOp;
    private Dictionary<string, (double Ms, long Calls)> _prevConv;
    private readonly OcrBackend _backend;

    public SharpEngine(string modelType, int workers, OcrBackend backend)
    {
        _backend = backend;
        _ocr = PaddleOcrAll.Load(BenchEngines.Bundle(modelType), new PaddleOcrOptions
        {
            LineWorkerCount = workers,
            Detector = new PaddleOcrDetectorOptions { MaxSessionCacheEntries = CacheEntries, Backend = backend },
            Recognizer = new PaddleOcrRecognizerOptions { AdaptiveWidth = true, TargetWidth = 320, Backend = backend },
            Classifier = new PaddleOcrClassifierOptions { Backend = backend },
        });
        PipelineProfiler.Enable(true);
        EngineStats.EnableProfiling(true);
        Extra["cacheEntries"] = CacheEntries;
        Extra["backend"] = backend.ToString().ToLowerInvariant();
        Extra["effectiveWorkers"] = _ocr.EffectiveLineWorkerCount;
        _workers = workers;
        _prevStages = PipelineProfiler.Snapshot();
        _prevOp = EngineStats.Operators();
        _prevConv = EngineStats.ConvClasses();
    }

    public string Name => "sharp";
    public JsonObject Extra { get; } = [];
    public string LoadedMessage(double workingSetMb) =>
        $"loaded working_set={workingSetMb:F1} MB engine=sharp backend={_backend.ToString().ToLowerInvariant()} workers={_ocr.EffectiveLineWorkerCount}/{_workers} cpu={Environment.ProcessorCount}";

    public BenchEngineOutput Run(byte[] bgr, int width, int height, int stride)
    {
        PaddleOcrResult result = _ocr.Run(bgr, width, height, stride);
        var cur = PipelineProfiler.Snapshot();
        var curOp = EngineStats.Operators();
        var curConv = EngineStats.ConvClasses();
        var stageMs = new Dictionary<string, double>();
        var stageCalls = new Dictionary<string, long>();
        for (int s = 0; s < PipelineProfiler.StageCount; s++)
        {
            stageMs[PipelineProfiler.StageNames[s]] = cur[s].Milliseconds - _prevStages[s].Milliseconds;
            stageCalls[PipelineProfiler.StageNames[s]] = cur[s].Calls - _prevStages[s].Calls;
        }
        var ops = Delta(curOp, _prevOp);
        var conv = Delta(curConv, _prevConv);
        _prevStages = cur;
        _prevOp = curOp;
        _prevConv = curConv;
        return new BenchEngineOutput
        {
            Detected = result.DetectedCount,
            Texts = result.Lines.Select(x => x.Text).ToArray(),
            Rotations = result.Lines.Select(x => x.AppliedRotationDegrees).ToArray(),
            Boxes = result.Lines.Select(x => BenchBoxes.Aabb(x.Box.X1, x.Box.Y1, x.Box.X2, x.Box.Y2,
                x.Box.X3, x.Box.Y3, x.Box.X4, x.Box.Y4)).ToArray(),
            Hash = $"{result.PackedTextHash:x16}",
            StageMs = stageMs,
            StageCalls = stageCalls,
            OperatorMs = ops,
            ConvClassMs = conv,
        };
    }

    public void Dispose() => _ocr.Dispose();

    private static Dictionary<string, BenchmarkMetric> Delta(Dictionary<string, (double Ms, long Calls)> cur,
        Dictionary<string, (double Ms, long Calls)> prev)
    {
        var d = new Dictionary<string, BenchmarkMetric>();
        foreach (var (name, (ms, calls)) in cur)
        {
            prev.TryGetValue(name, out var p);
            if (calls > p.Calls) d[name] = new BenchmarkMetric { Ms = ms - p.Ms, Calls = calls - p.Calls };
        }
        return d;
    }
}
