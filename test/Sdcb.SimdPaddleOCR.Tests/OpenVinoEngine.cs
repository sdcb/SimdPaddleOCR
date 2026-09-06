using System.Text;
using System.Text.Json.Nodes;
using OpenCvSharp;
using Sdcb.OpenVINO;
using Sdcb.OpenVINO.PaddleOCR;
using Sdcb.OpenVINO.PaddleOCR.Models;
using Sdcb.OpenVINO.PaddleOCR.Models.Details;
using Sdcb.SimdPaddleOCR.ModelProvider;
using OvPaddleOcrAll = Sdcb.OpenVINO.PaddleOCR.PaddleOcrAll;
using OvPaddleOcrOptions = Sdcb.OpenVINO.PaddleOCR.PaddleOcrOptions;
using OvPaddleOcrResult = Sdcb.OpenVINO.PaddleOCR.PaddleOcrResult;

namespace Sdcb.SimdPaddleOCR.Tests;

sealed class OpenVinoEngine : IBenchEngine
{
    private readonly string _tempDir;
    private readonly OvPaddleOcrAll _ocr;
    private bool _disposed;

    public OpenVinoEngine(string modelType)
    {
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("--engine openvino currently ships the Windows OpenVINO runtime");

        var bundle = BenchEngines.Bundle(modelType);
        _tempDir = Path.Combine(Path.GetTempPath(), "sdcb-simdpaddleocr-ov-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
        try
        {
            string detPath = WriteProvider(bundle.Detection, Path.Combine(_tempDir, "det.onnx"));
            string recPath = WriteProvider(bundle.Recognition, Path.Combine(_tempDir, "rec.onnx"));
            string[] labels = ReadLabels(bundle.Dictionary);
            ClassificationModel? cls = null;
            if (bundle.Classification is { } clsProvider)
            {
                string clsPath = WriteProvider(clsProvider, Path.Combine(_tempDir, "cls.onnx"));
                cls = new PathOnnxClassificationModel(clsPath);
            }

            var device = new DeviceOptions("CPU");
            _ocr = new OvPaddleOcrAll(
                new FullOcrModel(
                    new PathOnnxDetectionModel(detPath),
                    cls,
                    new PathOnnxRecognizationModel(recPath, labels)),
                new OvPaddleOcrOptions(device))
            {
                Enable180Classification = cls is not null,
                AllowRotateDetection = true,
                EnableDocumentOrientationClassification = false,
            };
            if (bundle.Detection.Version is "v6-tiny")
                _ocr.Detector.BoxScoreThreahold = 0.4f;
        }
        catch
        {
            TryDeleteTemp();
            throw;
        }

        Extra["ovInferenceNumThreads"] = "default";
        Extra["ovPerformanceMode"] = "LATENCY";
        Extra["ovDevice"] = "CPU";
    }

    public string Name => "openvino";
    public JsonObject Extra { get; } = [];
    public string LoadedMessage(double workingSetMb) =>
        $"loaded working_set={workingSetMb:F1} MB engine=openvino cpu={Environment.ProcessorCount}";

    public BenchEngineOutput Run(byte[] bgr, int width, int height, int stride)
    {
        if (stride != checked(width * 3))
            throw new ArgumentException("OpenVINO engine requires packed BGR (stride = width * 3).");
        using Mat src = Mat.FromPixelData(height, width, MatType.CV_8UC3, bgr);
        OvPaddleOcrResult result = _ocr.Run(src);
        string[] texts = result.Regions.Select(r => r.Text).ToArray();
        return new BenchEngineOutput
        {
            Detected = texts.Length,
            Texts = texts,
            Rotations = new int[texts.Length],
        };
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _ocr.Dispose();
        TryDeleteTemp();
    }

    private void TryDeleteTemp()
    {
        try
        {
            if (Directory.Exists(_tempDir))
                Directory.Delete(_tempDir, recursive: true);
        }
        catch
        {
            // temp cleanup is best-effort
        }
    }

    private static string WriteProvider(IPaddleOcrModelProvider provider, string path)
    {
        using Stream src = provider.OpenRead();
        using FileStream dst = File.Create(path);
        src.CopyTo(dst);
        return path;
    }

    private static string[] ReadLabels(IPaddleOcrModelProvider dictionary)
    {
        using Stream src = dictionary.OpenRead();
        using var reader = new StreamReader(src, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
        string text = reader.ReadToEnd();
        if (text.Length > 0 && text[0] == '\uFEFF') text = text[1..];
        string[] raw = text.Split('\n');
        if (raw.Length > 0 && raw[^1].Length == 0)
            raw = raw[..^1];
        return [.. raw.Select(x => x.EndsWith('\r') ? x[..^1] : x)];
    }

    sealed class PathOnnxDetectionModel(string path) : DetectionModel(ModelVersion.V6)
    {
        public override Model CreateOVModel(OVCore core) => core.ReadModel(path);
    }

    sealed class PathOnnxClassificationModel(string path) : ClassificationModel(ModelVersion.V6)
    {
        public override NCHW Shape => FileOnnxClassificationModel.DefaultTextLineOrientationShape;
        public override ClassificationPreprocessMode PreprocessMode => ClassificationPreprocessMode.ImageNetRgb;
        public override ClassificationResizeMode ResizeMode => ClassificationResizeMode.DirectResize;
        public override Model CreateOVModel(OVCore core) => core.ReadModel(path);
    }

    sealed class PathOnnxRecognizationModel(string path, IReadOnlyList<string> labels)
        : RecognizationModel(ModelVersion.V6)
    {
        public override Model CreateOVModel(OVCore core) => core.ReadModel(path);
        public override string GetLabelByIndex(int i) => GetLabelByIndex(i, labels);
    }
}
