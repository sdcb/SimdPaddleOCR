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
    private (long Ticks, long Calls)[] _prevOp;
    private (long Ticks, long Calls)[] _prevConv;

    public SharpEngine(string modelType, int workers)
    {
        _ocr = PaddleOcrAll.Load(BenchEngines.Bundle(modelType), new PaddleOcrOptions
        {
            LineWorkerCount = workers,
            Detector = new PaddleOcrDetectorOptions { MaxSessionCacheEntries = CacheEntries },
            Recognizer = new PaddleOcrRecognizerOptions { AdaptiveWidth = true, TargetWidth = 320 },
        });
        PipelineProfiler.Enable(true);
        InferenceSession.EnableProfiling(true);
        Extra["cacheEntries"] = CacheEntries;
        Extra["effectiveWorkers"] = _ocr.EffectiveLineWorkerCount;
        _workers = workers;
        _prevStages = PipelineProfiler.Snapshot();
        _prevOp = InferenceSession.ProfileSnapshot();
        _prevConv = InferenceSession.ConvClassProfileSnapshot();
    }

    public string Name => "sharp";
    public JsonObject Extra { get; } = [];
    public string LoadedMessage(double workingSetMb) =>
        $"loaded working_set={workingSetMb:F1} MB engine=sharp workers={_ocr.EffectiveLineWorkerCount}/{_workers} cpu={Environment.ProcessorCount}";

    public BenchEngineOutput Run(byte[] bgr, int width, int height, int stride)
    {
        PaddleOcrResult result = _ocr.Run(bgr, width, height, stride);
        var cur = PipelineProfiler.Snapshot();
        var curOp = InferenceSession.ProfileSnapshot();
        var curConv = InferenceSession.ConvClassProfileSnapshot();
        var stageMs = new Dictionary<string, double>();
        var stageCalls = new Dictionary<string, long>();
        for (int s = 0; s < PipelineProfiler.StageCount; s++)
        {
            stageMs[PipelineProfiler.StageNames[s]] = cur[s].Milliseconds - _prevStages[s].Milliseconds;
            stageCalls[PipelineProfiler.StageNames[s]] = cur[s].Calls - _prevStages[s].Calls;
        }
        var ops = new Dictionary<string, BenchmarkMetric>();
        string[] opNames = Enum.GetNames<OperatorId>();
        for (int op = 1; op < opNames.Length; op++)
        {
            double ms = (curOp[op].Ticks - _prevOp[op].Ticks) * 1000.0 / Stopwatch.Frequency;
            long calls = curOp[op].Calls - _prevOp[op].Calls;
            if (calls > 0)
                ops[opNames[op]] = new BenchmarkMetric { Ms = ms, Calls = calls };
        }
        var conv = new Dictionary<string, BenchmarkMetric>();
        string[] convNames = ["Conv1x1", "Conv3x3", "Depthwise3x3", "Stride2Conv3x3", "OtherConv"];
        for (int c = 0; c < 5; c++)
        {
            double ms = (curConv[c].Ticks - _prevConv[c].Ticks) * 1000.0 / Stopwatch.Frequency;
            long calls = curConv[c].Calls - _prevConv[c].Calls;
            if (calls > 0)
                conv[convNames[c]] = new BenchmarkMetric { Ms = ms, Calls = calls };
        }
        _prevStages = cur;
        _prevOp = curOp;
        _prevConv = curConv;
        return new BenchEngineOutput
        {
            Detected = result.DetectedCount,
            Texts = result.Lines.Select(x => x.Text).ToArray(),
            Rotations = result.Lines.Select(x => x.AppliedRotationDegrees).ToArray(),
            Hash = $"{result.PackedTextHash:x16}",
            StageMs = stageMs,
            StageCalls = stageCalls,
            OperatorMs = ops,
            ConvClassMs = conv,
        };
    }

    public void Dispose() => _ocr.Dispose();
}
