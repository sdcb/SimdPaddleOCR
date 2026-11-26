using System.Collections.Concurrent;
using System.Threading.Tasks;
using Sdcb.SimdPaddleOCR;

namespace SimdCodecs.AspNetCore;

public sealed class OcrEngine : IDisposable
{
    public const string DefaultModelName = "tiny";
    public static readonly string[] Models = ["tiny", "small", "medium"];

    private readonly OcrServerOptions _serverOptions;
    private readonly PaddleOcrOptions _ocrOptions;
    private readonly ConcurrentDictionary<string, Lazy<Task<PaddleOcrAll>>> _engines = new(StringComparer.Ordinal);

    public OcrEngine(OcrServerOptions serverOptions)
    {
        _serverOptions = serverOptions;
        DefaultModel = serverOptions.GetNormalizedDefaultModel();
        _ocrOptions = serverOptions.ToPaddleOcrOptions();
    }

    public string DefaultModel { get; }
    public long MaxImagePixels => _serverOptions.MaxImagePixels;

    public string[] GetLoadedModels() =>
    [
        .. _engines
            .Where(kv => kv.Value.IsValueCreated && kv.Value.Value.Status == TaskStatus.RanToCompletion)
            .Select(kv => kv.Key)
            .OrderBy(k => Array.IndexOf(Models, k))
    ];

    public bool TryNormalizeModel(string? model, out string name) =>
        TryNormalizeModel(model, DefaultModel, out name);

    public static bool TryNormalizeModel(string? model, string defaultModel, out string name)
    {
        name = string.IsNullOrWhiteSpace(model) ? defaultModel : model.Trim().ToLowerInvariant();
        return name is "tiny" or "small" or "medium";
    }

    public Task<PaddleOcrAll> GetAsync(string model)
    {
        if (!TryNormalizeModel(model, out string name))
            throw new ArgumentOutOfRangeException(nameof(model), "模型必须是 tiny、small 或 medium。");

        return _engines.GetOrAdd(name, key => new Lazy<Task<PaddleOcrAll>>(
            () => LoadModelAsync(key, _ocrOptions),
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

    private static Task<PaddleOcrAll> LoadModelAsync(string name, PaddleOcrOptions options) => name switch
    {
        "tiny" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.ChineseV6TinyModels.Default, options),
        "small" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Small.ChineseV6SmallModels.Default, options),
        "medium" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Medium.ChineseV6MediumModels.Default, options),
        _ => throw new ArgumentOutOfRangeException(nameof(name))
    };
}

internal sealed class OcrPreloadHostedService(
    OcrEngine engine,
    OcrServerOptions options,
    ILogger<OcrPreloadHostedService> logger) : IHostedService
{
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        string[] models = options.ResolvePreloadModels();
        foreach (string model in models)
        {
            cancellationToken.ThrowIfCancellationRequested();
            logger.LogInformation("Preloading OCR model '{Model}'...", model);
            await engine.GetAsync(model).WaitAsync(cancellationToken);
            logger.LogInformation("OCR model '{Model}' loaded.", model);
        }
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
