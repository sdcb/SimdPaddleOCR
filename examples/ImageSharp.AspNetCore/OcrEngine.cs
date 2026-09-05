using System.Collections.Concurrent;
using System.Threading.Tasks;
using Sdcb.SimdPaddleOCR;

namespace ImageSharp.AspNetCore;

public sealed class OcrEngine : IDisposable
{
    public const long MaxImagePixels = 40_000_000;
    public const string DefaultModel = "tiny";
    public static readonly string[] Models = ["tiny", "small", "medium"];

    private readonly ConcurrentDictionary<string, Lazy<Task<PaddleOcrAll>>> _engines = new(StringComparer.Ordinal);

    public static bool TryNormalizeModel(string? model, out string name)
    {
        name = string.IsNullOrWhiteSpace(model) ? DefaultModel : model.Trim().ToLowerInvariant();
        return name is "tiny" or "small" or "medium";
    }

    public Task<PaddleOcrAll> GetAsync(string model)
    {
        if (!TryNormalizeModel(model, out string name))
            throw new ArgumentOutOfRangeException(nameof(model), "模型必须是 tiny、small 或 medium。");

        return _engines.GetOrAdd(name, static key => new Lazy<Task<PaddleOcrAll>>(
            () => LoadModelAsync(key),
            LazyThreadSafetyMode.ExecutionAndPublication)).Value;
    }

    public void Dispose()
    {
        foreach (Lazy<Task<PaddleOcrAll>> lazy in _engines.Values)
        {
            if (!lazy.IsValueCreated)
                continue;

            try
            {
                if (lazy.Value.Status == TaskStatus.RanToCompletion)
                    lazy.Value.Result.Dispose();
            }
            catch
            {
            }
        }

        _engines.Clear();
    }

    private static Task<PaddleOcrAll> LoadModelAsync(string name) => name switch
    {
        "tiny" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.ChineseV6TinyModels.Default),
        "small" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Small.ChineseV6SmallModels.Default),
        "medium" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Medium.ChineseV6MediumModels.Default),
        _ => throw new ArgumentOutOfRangeException(nameof(name))
    };
}
