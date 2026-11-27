namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>Model / dataset locations and image decoding shared by the phone
/// host and the desktop GpuBench (which links this file).</summary>
static class Paths
{
    internal static string Root = "";
    internal static string ModelsDir = "models";
    internal static string DatasetDir = "dataset";

    internal static string Path_(string p) => Path.IsPathRooted(p) ? p : Path.Combine(Root, p);
    internal static string ModelPath(string model, string kind) =>
        kind == "cls" ? Path.Combine(Path_(ModelsDir), "cls.onnx") : Path.Combine(Path_(ModelsDir), $"{model}-{kind}.onnx");
    internal static string DictPath(string model) => Path.Combine(Path_(ModelsDir), $"{model}-dict.txt");
    internal static string Dataset => Path_(DatasetDir);

    internal static OcrBackend ParseBackend(string s) => s.ToLowerInvariant() switch
    {
        "cpu" or "sharp" => OcrBackend.Cpu,
        "vulkan" => OcrBackend.Vulkan,
        "auto" => OcrBackend.Auto,
        _ => throw new ArgumentException("backend must be cpu, vulkan or auto"),
    };

    /// <summary>Decodes like the desktop Tests harness (SimdCodecs, packed BGR24).</summary>
    internal static (byte[] Bgr, int W, int H) LoadBgr(string path)
    {
        using var reader = Sdcb.SimdCodecs.ImageCodecs.OpenRowReader(File.OpenRead(path), Sdcb.SimdCodecs.Abstractions.PixelFormat.Bgr24);
        int stride = checked(reader.Width * 3);
        byte[] pixels = new byte[checked(stride * reader.Height)];
        int y = 0;
        while (y < reader.Height)
            y += reader.ReadRows(pixels.AsSpan(y * stride), reader.Height - y, stride);
        return (pixels, reader.Width, reader.Height);
    }

    internal static string[] DatasetFiles() => Directory.GetFiles(Dataset, "img-*.jpg")
        .OrderBy(x => x, StringComparer.OrdinalIgnoreCase).ToArray();
}
