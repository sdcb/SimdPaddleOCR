using System.Text.Json.Nodes;
using LwPpocrCSharp;

namespace Sdcb.SimdPaddleOCR.Tests;

sealed class CEngine : IBenchEngine
{
    private readonly NativeOcr _ocr;

    public CEngine(string cAssetsDir, int workers, string modelType)
    {
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("--engine c requires Windows (lw_ppocr_c.dll)");
        if (modelType != "tiny")
            throw new ArgumentException("--engine c only supports --model tiny");

        CAssets.EnsureAsync(cAssetsDir).GetAwaiter().GetResult();
        CAssets.CopyDll(cAssetsDir);
        string dictPath = CAssets.WriteDictionary(cAssetsDir);
        Extra["cAssets"] = cAssetsDir;
        Extra["cDll"] = CAssets.BaseUrl + CAssets.DllName;
        Extra["cDet"] = CAssets.BaseUrl + CAssets.DetName;
        Extra["cCls"] = CAssets.BaseUrl + CAssets.ClsName;
        Extra["cRec"] = CAssets.BaseUrl + CAssets.RecName;
        Extra["cDict"] = dictPath;

        _ocr = new NativeOcr(
            CAssets.DetPath(cAssetsDir),
            CAssets.ClsPath(cAssetsDir),
            CAssets.RecPath(cAssetsDir),
            dictPath,
            useDirectionClassification: true,
            (uint)workers);
    }

    public string Name => "c";
    public JsonObject Extra { get; } = [];
    public string LoadedMessage(double workingSetMb) =>
        $"loaded working_set={workingSetMb:F1} MB engine=c";

    public BenchEngineOutput Run(byte[] bgr, int width, int height, int stride)
    {
        OcrResponse result = _ocr.RecognizeDecoded(new DecodedBgrImage
        {
            Pixels = bgr,
            Width = width,
            Height = height,
            Stride = stride,
        });
        return new BenchEngineOutput
        {
            Detected = result.detected_count,
            Texts = result.result.Select(x => x.text).ToArray(),
            Rotations = result.result.Select(x => x.rotation).ToArray(),
        };
    }

    public void Dispose() => _ocr.Dispose();
}
